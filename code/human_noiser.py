# This class introduces human-originated noise into a dataset, using individual
# annotations as a noise source, as described in:
# https://nlp.stanford.edu/pubs/chong2022labelerrors.pdf

import csv, random, json

class BaseNoiser:

    def noise_label_files(self, data_cfg, txt_filelist, lbl_filelist, noise_seed=42, noise_rate=0.05, custom_params=None):
        """
        Fixed interface for all noisers.
        custom_params format is defined per noiser

        txt_filelist: List of fulltext files
        lbl_filelist: List of label files
        noise_seed: The random seed to use for noise generation
        noise_rate: The percent of labels to noise
        """
        raise NotImplementedError

    def get_all_labels(self, lbl_filelist):
        """
        Looks through a list of label files and returns a list of strings of all
        possible labels, and their counts
        """
        all_labels = {}

        for lblfile in lbl_filelist:
            with open(lblfile, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['label'] not in all_labels:
                        all_labels[row['label']] = 1
                    else:
                        all_labels[row['label']] += 1

        return all_labels



class AnnotationNoiser(BaseNoiser):
    """
    Noiser which uses human annotations provided within fulltext files as noise
    source.

    Noise types: We allow for three types of noise:
    - hard: Random label errors where a mininum proportion of annotators
        all made the same error (which doesn't match the gold label)
        Simulates a systematic error made by lots of annotators
    - rogue: Random label errors coming from a single annotator
        Requires dataconfig.annotator_ids_provided = True
        Simulates an annotator whose input was not verified at all
    - malicious: Particularly rare annotations:
        Simulates an actively harmful labeler
        Swaps labels to an annotation label selected by exactly one annotator
    - missed: Random label errors coming from all annotators
        Simulates a random careless mistake that was not caught by verifiers

    Interface:
    - noise_rate is the top-line noise rate: the total of all three of the above
    - custom_params is a json string with the following format (with no spaces)
        - [hard_proportion, rogue_proportion, malicious_proportion, 
           missed_proportion, hard_threshold, die_on_insufficient]
        - the proportions must sum to 1.0
        - difficulty threshold values must be [0.0,1.0]
        - die_on_insufficient is a boolean
    - Defaults to 100% missed if not provided

    Example: on a dataset of 10000 items with 5 annotations, 
    0.05 [0.4,0.3,0.2,0.1,0.6,true] would produce:
    - 500 mislabelings total, out of which
    - 200 are hard, such that >=3/5 annotators made the same error
    - 150 are rogue, from one specific annotator
    - 100 are malicious, minority annotations
    - 50 are missed, from all annotators
    - Fails if there aren't enough of any kind

    Order: Noising is done in order: hard -> rogue -> malicious -> missed,
    because there are fewer of the former than the latter, so starting with
    missed might result in not being able to get enough challenging

    Files: Each label file is noised to the target rate separately, because if
    we don't do this, there will be 1-2% variance between noise rates by file.

    Insufficiency: Unless we're trying to maximally noise, it's better to die
    on insufficient mislabelings, or we don't have good comparability
    """

    @staticmethod
    def noise_data(row_data, annotations_by_idx, annotations_per_row,
        filename, candidate_idxs, n_to_noise, 
        disagreement_range, only_consider_majority, die_if_insufficient):
        """
        Key noising helper.

        Perform noising operation on a subset of candidate idxs, within the
        entire dataset. This function can be used for a bunch of different
        annotator-based noising schemes.

        - row_data: {idx:{row_data}}: edit 'label' field to introduce noise
        - annotations_by_idx: {idx: {ann_label: count}}: helper data
        - annotations_per_row: self-evident (SNLI has 5)
        - filename: provided for logging
        - candidate_idxs: [idx]: which rows are currently eligible for noising
        - n_to_noise: as implied 
        - disagreement_range: [low,high]: labels are only
            eligible for noising with a value that represents [low,high]*100% of
            a given idx's annotations
        - only_consider_majority: if True, only check majority label against
            disagreement range. Otherwise any annotator label within the range
            is eligible for noising.
        - die_if_insufficient: raise exception the second we fail to meet noise 
            target. We might choose not to do this if we were setting entire
            dataset to some status (e.g.: majority vote)

        Noising is done at random, but all mislabelings are put into a pool
        with one entry for every suitable non-gold annotation, because IRL,
        a row with [G,G,G,G,NG] is less likely to accidentally come mislabeled
        than a row with [G,NG,NG,NG,NG].

        If a label is already noised, it won't be noised again.
        """

        # sanity check
        assert(disagreement_range[0] >= 0.0 and disagreement_range[0] <= 1.0)
        assert(disagreement_range[1] >= 0.0 and disagreement_range[1] <= 1.0)
        assert(disagreement_range[0] <= disagreement_range[1])
        assert(n_to_noise >= 0 and n_to_noise <= len(candidate_idxs))

        min_annotations = disagreement_range[0] * annotations_per_row
        max_annotations = disagreement_range[1] * annotations_per_row

        # prep: make a pool of suitable annotations which we can sample from
        # it's better to do this upfront so that:
        # - sprinkled noise is sampled in proportion to its frequency
        #   (e.g.: 1/5 err + 4/5 gold = only one lottery ticket)
        # - we can tell if we have enough right away

        mislabeling_pool = [] # [(idx,ann_label,count),]
        n_max_noiseable = 0
        for idx in candidate_idxs:
            if only_consider_majority:
                # sort annotations by count, and take the top non-gold one 
                # (if there's a 2-2-1 tie this works out random)
                ann_sorted = sorted(annotations_by_idx[idx].items(), key=lambda x: x[1], reverse=True)
                # if the first one is gold, pop it
                if ann_sorted[0][0] == row_data[idx]['label']:
                    ann_sorted = ann_sorted[1:]
                pool_cands = ann_sorted[0:1]
            else: # take all non-gold annotations                
                pool_cands = [a for a in annotations_by_idx[idx].items() \
                    if a[0] != row_data[idx]['label']]

            # filter out annotations by disagreement range
            pool_cands = [a for a in pool_cands \
                if a[1] >= min_annotations and a[1] <= max_annotations]

            if len(pool_cands) > 0:
                n_max_noiseable += 1

            # add one entry to pool for each count
            for ann_label,count in pool_cands:
                for _ in range(count):
                    mislabeling_pool.append((idx,ann_label,count))

        # sanity check
        if n_max_noiseable < n_to_noise:
            if die_if_insufficient:
                raise Exception(f"Noiser: Insufficient noiseable rows: {filename}: {n_max_noiseable} noiseable < {n_to_noise} desired, at disagreement range of {disagreement_range}")

        # shuffle noising pool and work through it until we hit our target
        random.shuffle(mislabeling_pool)
        n_noised = 0
        for idx,ann_label,count in mislabeling_pool:
            if n_noised >= n_to_noise:
                break # end if hit noise target
            if row_data[idx]['label'] != row_data[idx]['label_clean']:
                continue # skip if already noised

            # apply noise
            row_data[idx]['label'] = ann_label
            n_noised += 1

        # done: sanity check: have we reached noise target?
        print(f"\tNoiser: {filename}: {n_noised} noised / {n_to_noise} target / {n_max_noiseable} possible / {len(candidate_idxs)} rows: {n_noised/len(candidate_idxs)*100:2f} percent")

        return n_noised


    def noise_label_files(self, data_cfg, txt_filelist, lbl_filelist, noise_seed=42, noise_rate=0.05, custom_params=None):
        """
        The standard interface function
        """
        random.seed(noise_seed)

        # parse custom parameters
        if custom_params is None or len(custom_params) == 0:
            hard_rate, rogue_rate, adv_rate, missed_rate, hard_threshold, die_if_insufficient = 0,0,0,1,0,True
        else:
            cparams = json.loads(custom_params)
            assert(len(cparams)==6)
            hard_rate, rogue_rate, adv_rate, missed_rate, hard_threshold, die_if_insufficient = cparams
        assert(missed_rate >= 0 and missed_rate <= 1)
        assert(rogue_rate >= 0 and rogue_rate <= 1)
        assert(adv_rate >= 0 and adv_rate <= 1)
        assert(hard_rate >= 0 and hard_rate <= 1)
        assert(hard_threshold >= 0 and hard_threshold <= 1)
        assert(abs(hard_rate + rogue_rate + adv_rate + missed_rate - 1) < 1e-5)

        # preindex all label files, by idx and filename
        # noising edits are made by editing row data in row_by_idx
        row_by_idx = {} # holds {idx:{row_data}}
        idxs_by_file = {} # {filename:[idx]}
        files_by_idx = {}
        for lblfile in lbl_filelist:
            idxs_by_file[lblfile] = []
            with open(lblfile, 'r') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    row_by_idx[row['idx']] = row
                    idxs_by_file[lblfile].append(row['idx'])
                    if row['idx'] in files_by_idx: # dupe idx's confuse noising
                        raise Exception(f"Duplicate idx in label files: {row['idx']}")
                    files_by_idx[row['idx']] = lblfile


        # preindex all annotators
        annotations = [] # [(annotator, {idx:ann_label}),]
        annotations_by_idx = {} # {idx: {ann_label: count}}
        for txtfile in txt_filelist:
            with open(txtfile, 'r') as f:
                reader = csv.DictReader(f)
                fieldnames_an = reader.fieldnames
                if len(annotations) == 0: # init index
                    for i in range(data_cfg.annotator_labels_per_row):
                        annotations.append((i,dict()))
                for row in reader:
                    idx = row['idx']
                    annotations_by_idx[idx] = {}
                    for i in range(data_cfg.annotator_labels_per_row):
                        ann_label = row[f'label_a{i}']
                        if ann_label != "":
                            annotations[i][1][idx] = ann_label
                            if ann_label not in annotations_by_idx[idx]:
                                annotations_by_idx[idx][ann_label] = 1
                            else:
                                annotations_by_idx[idx][ann_label] += 1

        # first we save original gold labels into label_clean column
        for row in row_by_idx.values():
            row['label_clean'] = row['label']

        # calculate number of mislabelings desired

        # calculate number of mislabelings to target for each file
        n_noise_counts = {}
        for lbl_file in lbl_filelist:
            n_noise_counts[lbl_file] = {
                'missed': [0,round(len(idxs_by_file[lbl_file])*noise_rate*missed_rate)], 
                'rogue': [0,round(len(idxs_by_file[lbl_file])*noise_rate*rogue_rate)],
                'adv': [0,round(len(idxs_by_file[lbl_file])*noise_rate*adv_rate)],
                'hard': [0,round(len(idxs_by_file[lbl_file])*noise_rate*hard_rate)]
            }

        print("Applying Hard noising...")
        for lbl_file in lbl_filelist: # file by file
            n_noise_counts[lbl_file]['hard'][0] = self.noise_data(
                row_data=row_by_idx, 
                annotations_by_idx=annotations_by_idx, 
                annotations_per_row=data_cfg.annotator_labels_per_row,
                filename=lbl_file,
                candidate_idxs=idxs_by_file[lbl_file],
                n_to_noise=n_noise_counts[lbl_file]['hard'][1], 
                disagreement_range=[hard_threshold,1],
                only_consider_majority=True,
                die_if_insufficient=die_if_insufficient,
            )

        # Rogue noising: Apply annotator by annotator, until all files hit target noise rate
        print("Applying Rogue noising...")
        n_noise_rogue  = int(len(row_by_idx)*noise_rate*rogue_rate)
        n_noised_rogue, n_files_noised_rogue = 0, 0
        if n_noise_rogue > 0:
            if data_cfg.annotations_anonymized == True:
                raise Exception(f"Rogue annotations are not valid for datasets with anonymised labels.")
            random.shuffle(annotations)
            for ann_no,ann_data in annotations:
                print(f"Rogue noise: Applying a{ann_no}'s annotations with {n_noised_rogue}/{n_noise_rogue} currently noised")
                ann_idxs = list(ann_data.keys())
                random.shuffle(ann_idxs)
                for idx in ann_idxs:
                    # only apply to a file if it's not over its noising limit
                    # (and not already noised)
                    to_file = files_by_idx[idx]
                    if n_noise_counts[to_file]['rogue'][0] < n_noise_counts[to_file]['rogue'][1]:
                        if row_by_idx[idx]['label'] == row_by_idx[idx]['label_clean'] and \
                            row_by_idx[idx]['label'] != ann_data[idx]:

                            row_by_idx[idx]['label'] = ann_data[idx]
                            n_noise_counts[to_file]['rogue'][0] += 1
                            n_noised_rogue += 1
                            if n_noise_counts[to_file]['rogue'][0] >= n_noise_counts[to_file]['rogue'][1]:
                                n_files_noised_rogue += 1

                if n_files_noised_rogue == len(lbl_filelist):
                    print(f"Rogue noise: Requirements met: {n_noised_rogue} rows noised!")
                    break
        if n_noised_rogue < n_noise_rogue:
            print(f"Rogue Noise: Ran all annotators and did not meet noise requirement {n_noise_rogue}: {n_noised_rogue} applied")
            if die_if_insufficient:
                raise Exception(f"Rogue noise: Not enough rows noised! {n_noised_rogue}/{n_noise_rogue}")
        for lbl_file in lbl_filelist:
            noised, target = n_noise_counts[lbl_file]['rogue']
            n_rows = len(idxs_by_file[lbl_file])
            print(f"\tRogue Noise: {lbl_file}: {noised} noised / {target} target / {n_rows} rows: {noised/n_rows*100:2f} percent")

        print("Applying Malicious noising...")
        one_labeler_percent = 1/data_cfg.annotator_labels_per_row
        for lbl_file in lbl_filelist: # file by file
            n_noise_counts[lbl_file]['missed'][0] = self.noise_data(
                row_data=row_by_idx, 
                annotations_by_idx=annotations_by_idx, 
                annotations_per_row=data_cfg.annotator_labels_per_row,
                filename=lbl_file,
                candidate_idxs=idxs_by_file[lbl_file],
                n_to_noise=n_noise_counts[lbl_file]['adv'][1], 
                disagreement_range=[0,one_labeler_percent],
                only_consider_majority=False,
                die_if_insufficient=die_if_insufficient,
            )

        print("Applying Missed noising...")
        for lbl_file in lbl_filelist: # file by file
            n_noise_counts[lbl_file]['missed'][0] = self.noise_data(
                row_data=row_by_idx, 
                annotations_by_idx=annotations_by_idx, 
                annotations_per_row=data_cfg.annotator_labels_per_row,
                filename=lbl_file,
                candidate_idxs=idxs_by_file[lbl_file],
                n_to_noise=n_noise_counts[lbl_file]['missed'][1], 
                disagreement_range=[0,1],
                only_consider_majority=False,
                die_if_insufficient=die_if_insufficient,
            )

        # write data back to CSVs
        fieldnames.append('label_clean')
        for lblfile in lbl_filelist:
            with open(lblfile, 'w') as f:
                writer = csv.DictWriter(f, fieldnames)
                writer.writeheader()
                for idx in idxs_by_file[lblfile]:
                    writer.writerow(row_by_idx[idx])

        return


