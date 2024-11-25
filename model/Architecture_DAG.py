[Input Image]
       |
[Convolutional Layers]
       |
[Pool3 (Downsampling)]------------------------------+
       |                                            |
[Convolutional Layers]                              |
       |                                            |
[Pool4 (Downsampling)]-------------------------+    |
       |                                       |    |
[Convolutional Layers]                         |    |
       |                                       |    |
[Pool5 (Downsampling)]                         |    |
       |                                       |    |
[Score_fr (1x1 Conv)]                          |    |
       |                                       |    |
[Upscore_pool5 (2x Upsampling)]                |    | 
       |                                       |    |
       +---> [Score_Pool4 (1x1 Conv)] -------> +    |
                      |                             |
            [Element-wise Addition]                 |
                      |                             |
            [Upscore_pool4 (x2 Upsampling)]         |
                      |                             |
                      |                             |
            [Score Pool3 (1x1 Conv)]<---------------+
                      |
            [Element-wise Addition]
                      |
            [Upscore_pool3 (x8 Upsampling)]
                      |
                  [Output]
                      |
            [Per-pixel Class Scores]
