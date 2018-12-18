# Cross-Language Duplicate Detection
Duplicate Detection aims at identifying duplicate records across datasets. Cross-Language Duplicate Detection (CLDD) enables users to link records from datasets in different languages.



## Content

### dataset_extraction
This folder contains Python files to extract datasets from DBPedia infobox files and Article title files. It also includes blocking and labeling using interlanguage links provided by DBPedia.

### cldd
This folder contains the files that manages feature extraction and OOV treatment. The features include cross-language word embedding features and multilingual lexical knowledge base features

### uwn_java
This folder contains the Java file to extract concepts from UWN Java API.

### tests
This folder contains test files to compare our approach with baselines and measure performance of different components (features, OOV treatment, classifier, word embedding models).


