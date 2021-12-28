#### This folder stores the data processing script and processed data.

```
.
└── data
    └── VAD.json  # this json file stores the NRC_VAD vectors
    └── EMO_INT.json  # this json file stores words and their emotion intensity values between 0 and 1, which are calculated from VAD.json. The higher the value, the stronger the emotion intensity.  
    └── ConceptNet.json  # this json file stores the ConceptNet commonsense tuples.
    └── kemp_dataset_preproc.json  # the processed data used for training our model KEMP.
    └── EmpatheticDialogue/dataset_preproc.json  # EmpatheticDialogue dataset saved in json format
```