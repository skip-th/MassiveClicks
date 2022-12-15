import os

class read_categories:
    def __init__(self):
        self.file = None

    def set_file(self, file_path):
        self.file = open(file_path, 'r')

    def get_value(self, category):
        if category == 'iteration time':
            return self._get_iteration_time()
        elif category == 'computation time':
            return self._get_computation_time()
        elif category == 'update time':
            return self._get_update_time()
        elif category == 'synchronization time':
            return self._get_synchronization_time()
        elif category == 'training time':
            return self._get_training_time()
        elif category == 'evaluation time':
            return self._get_evaluation_time()
        elif category == 'parsing time':
            return self._get_parsing_time()
        elif category == 'h2d time':
            return self._get_h2d_time()
        elif category == 'd2h time':
            return self._get_d2h_time()
        elif category == 'perplexity':
            return self._get_perplexity()
        elif category == 'log likelihood':
            return self._get_log_likelihood_time()

    def _get_iteration_time(self):
        matches = []
        for line in self.file.readlines():
            if "Average time per iteration: " in line:
                matches.append(float(line.split("Average time per iteration: ")[-1].strip()))
        return matches

    def _get_computation_time(self):
        matches = []
        for line in self.file.readlines():
            if "Average time per computation in each Iteration: " in line:
                matches.append(float(line.split("Average time per computation in each Iteration: ")[-1].strip()))
        return matches

    def _get_update_time(self):
        matches = []
        for line in self.file.readlines():
            if "Average time per update in each Iteration: " in line:
                matches.append(float(line.split("Average time per update in each Iteration: ")[-1].strip()))
        return matches

    def _get_synchronization_time(self):
        matches = []
        for line in self.file.readlines():
            if "Average time per synchronization in each Iteration: " in line:
                matches.append(float(line.split("Average time per synchronization in each Iteration: ")[-1].strip()))
        return matches

    def _get_training_time(self):
        matches = []
        for line in self.file.readlines():
            if "Total time of training: " in line:
                matches.append(float(line.split("Total time of training: ")[-1].strip()))
        return matches

    def _get_evaluation_time(self):
        matches = []
        for line in self.file.readlines():
            if "Evaluation time: " in line:
                matches.append(float(line.split("Evaluation time: ")[-1].strip()))
        return matches

    def _get_parsing_time(self):
        matches = []
        for line in self.file.readlines():
            if "Total pre-processing time: " in line:
                matches.append(float((line.split("Total pre-processing time: ")[-1]).split(' seconds,')[0].strip()))
        return matches

    def _get_h2d_time(self):
        matches = []
        for line in self.file.readlines():
            if "Average Host to Device parameter transfer time: " in line:
                matches.append(float(line.split("Average Host to Device parameter transfer time: ")[-1].strip()))
        return matches

    def _get_d2h_time(self):
        matches = []
        for line in self.file.readlines():
            if "Average Device to Host parameter transfer time: " in line:
                matches.append(float(line.split("Average Device to Host parameter transfer time: ")[-1].strip()))
        return matches

    def _get_perplexity(self):
        matches = []
        for line in self.file.readlines():
            if "Perplexity is: " in line:
                matches.append(float(line.split("Perplexity is: ")[-1].strip()))
        return matches

    def _get_log_likelihood_time(self):
        matches = []
        for line in self.file.readlines():
            if "Total Log likelihood is: " in line:
                matches.append(float(line.split("Total Log likelihood is: ")[-1].strip()))
        return matches

class read_dataset:
    def __init__(self):
        self.data = dict()
        self.categories = [('iteration time', None), ('computation time', None), ('update time', None), ('synchronization time', None),
                           ('training time', None), ('evaluation time', None), ('parsing time', None),
                           ('h2d time', None), ('d2h time', None),
                           ('perplexity', None), ('log likelihood', None)]
        file_paths = self._get_relative_paths()
        self._make_data_structure(file_paths)
        self._fill_data_structure(self.data)

    def _get_relative_paths(self, directory=os.getcwd(), extensions=['txt']):
        file_paths = []
        for dirpath,_,filenames in os.walk(directory):
            for f in filenames:
                if f.rsplit('.', 1)[-1] in extensions:
                    file_paths.append(os.path.relpath(os.path.join(dirpath, f)))
        return file_paths


    def _make_data_structure(self, file_paths):
        for item in file_paths:
            p = self.data
            for x in item.split('/'):
                if '.' not in x:
                    p = p.setdefault(x, {})
                else:
                    p['file'] = item
                    for category, default_value in self.categories:
                        p[category] = default_value


    def _fill_data_structure(self, data_structure, keys=[]):
        dataset_categories = read_categories()
        for key, value in data_structure.items():
            if isinstance(value, dict):
                keys.append(key)
                self._fill_data_structure(value, keys)
            else:
                file_path = data_structure['file']
                for category, _ in self.categories:
                    dataset_categories.set_file(file_path)
                    data_structure[category] = dataset_categories.get_value(category)

    def get_dataset(self):
        return self.data

    def get_categories(self):
        return self.categories