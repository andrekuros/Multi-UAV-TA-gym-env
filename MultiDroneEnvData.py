import numpy as np


class sceneData:
    def __init__(self):
        
        # Tamanho da tela        
        self.GameArea = [1200,700]
        self.ContactLine = 550
        self.Bases = [np.array([400,680])]
        
        self.UavTypes = ["F1", "F2", "R1", "R2", "C1", "C2"]
                  
        self.TaskTypes =                    [ "Att"  , "Jam" , "Esc" , "Rec" , "Com" ]
        self.UavCapTable ={ "F1" : np.array([   1.0  ,  0.3  ,  1.0  ,  0.2  ,  0.5  ]),
                            "F2" : np.array([   0.7  ,  0.5  ,  0.8  ,  0.2  ,  0.5  ]),
                            "R1" : np.array([   0.0  ,  0.0  ,  0.0  ,  1.0  ,  0.5  ]),
                            "R2" : np.array([   0.0  ,  0.3  ,  0.0  ,  1.0  ,  1.0  ]),
                            "C1" : np.array([   0.0  ,  1.0  ,  0.0  ,  0.0  ,  0.8  ]),
                            "C2" : np.array([   0.0  ,  0.7  ,  0.0  ,  0.0  ,  1.0  ])}
                
        self.UavIndex =  {name : idx  for idx, name in enumerate(self.UavTypes)}
        self.TaskIndex = {name : idx  for idx, name in enumerate(self.TaskTypes)}
        
        self.maxSpeeds = {  "F1" : 7.0,
                            "F2" : 7.0,
                            "R1" : 4.0,
                            "R2" : 3.5,
                            "C1" : 2.0,
                            "C2" : 1.5}
        
        self.Endurances = { "F1" : 1000,
                            "F2" : 800,
                            "R1" : 2000,
                            "R2" : 2500,
                            "C1" : 3500,
                            "C2" : 4000}
        
        self.relayArea = {  "F1" : 20.0,
                            "F2" : 20.0,
                            "R1" : 50.0,
                            "R2" : 40.0,
                            "C1" : 200.0,
                            "C2" : 250.0}
        
        self.failTable = {  "F1" : 1.0,
                            "F2" : 1.2,
                            "R1" : 0.8,
                            "R2" : 1.0,
                            "C1" : 1.5,
                            "C2" : 2.0}
        
        
    def getTaskDuration(self, task_type):
        
        expectedDurations = {"Att" :  50,
                             "Jam" : 100,
                             "Esc" : 100,
                             "Rec" : 100,
                             "Com" : 100,
                             }
        
        return expectedDurations[task_type]