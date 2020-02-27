# object-gaming
A Game controller using Object-Detection **under progress*

**File Tree:**     

object_detection/
      ├── *main.py*
      |
      ├── data/
      │    ├── images/
      │    │      └── ...
      │    ├── annotations/
      │    │      └── ...
      │    ├── train_labels/
      │    │      └── ...
      │    ├── test_labels/
      │    │      └── ...
      │    ├── label_map.pbtxt
      │    ├── test_labels.csv
      │    ├── train_labels.csv
      │    ├── test_labels.records
      │    └── train_labels.records
      │
      └── models/           
           ├─ research/   
           │    ├── object_detection/
           │    │      ├── utils/
           |    |      |     ├── *visualization_utils.py*
           |    |      |                              
           │    │      ├── *game_utils.py*
           │    │      ├── *webcam_interface.py*
           │    │      └── ...
           │    │                       
           │    └── ...
           └── ...
