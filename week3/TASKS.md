Okay, this is how I see it. In brackets we can write the asigness. Once you are done with a task, ~~cross it out~~:

0. T0 Make sure detectron 2 correctly installed on cluster [Juan]

    1. Begginners tutorial or similar runs on cluster 

1. T1 - Load both datasets correctly [*asignee*]

    1. Function for loading MOTS dataset correctly is created (Note: pedestrian Id =2)

    1. Same for KITTI-MOTS. Note we want car id = 1 

    1. Dataset description is added to the overleaf project 

    > This task kindof blocks the next two

1. T2 - Evaluate pre-trained models
    1. Qualitative [*asignee*]
        1. Get example inferences, analyse success cases and failed cases. See sample images with detection and confidence drawn over it. For:
            - Fast RCNN
            - RetineNet
        1. Add discussion, images, etc to Overleaf
    1. <a name="test"> Quantitative: </a> [*asignee*]
        1. AP metrics can be obtained using detectrons 2 COCO Evaluator. Some work on deciding what data to use to train/test, map labels correctly... etc. This task can hopefully be reused to evaluate our training in T3.1
         1. Add discussion, images, etc to Overleaf

1. T3 - Train our own model [*asignee*]
    1. A module for training a model using all available data is created. Make sure it is easy to change the configuration (and ideally, run many tests in loop) 
    1. Reuse and adapt [T2.2](#test) to get numerical results. Once again, the easier it is to run experiments and get a table-like comparison the better.
