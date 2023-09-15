import numpy as np


class SceneData:
    def __init__(self):
        
        # Tamanho da tela        
        self.GameArea = [1200,700] #each pixel represent 0.5nm
        self.ContactLine = 550
        self.Bases = [np.array([400,680])]
        
        self.UavTypes = [ "R1", "R2", "F1", "F2", "C1", "C2"]
                                         
        self.TaskTypes =                    [ "Idle" , "Rec"  , "Att" , "Esc" , "Jam" , "Com" ]
        self.UavCapTable ={ "R1" : np.array([   0.0  ,  1.0  ,  0.0  ,  0.0  ,  0.5  ,  0.5  ]),
                            "R2" : np.array([   0.0  ,  0.8  ,  0.0  ,  0.0  ,  0.3  ,  1.0  ]),
                            "F1" : np.array([   0.0  ,  0.2  ,  1.0  ,  1.0  ,  0.3  ,  0.5  ]),
                            "F2" : np.array([   0.0  ,  0.3  ,  0.85  ,  0.8  ,  0.5  , 0.5  ]),                            
                            "C1" : np.array([   0.0  ,  0.0  ,  0.0  ,  0.0  ,  1.0  ,  0.8  ]),
                            "C2" : np.array([   0.0  ,  0.0  ,  0.0  ,  0.0  ,  0.7  ,  1.0  ])}
        
        self.UavCapTableByIdx = np.array([self.UavCapTable[ut] for ut in self.UavTypes])
        
        self.UavIndex =  {name : idx  for idx, name in enumerate(self.UavTypes)}
        self.TaskIndex = {name : idx  for idx, name in enumerate(self.TaskTypes)}
        
        self.maxSpeeds = {  "F1" : 20.0,
                            "F2" : 15.0,
                            "R1" : 10.0,
                            "R2" : 15.0,
                            "C1" : 5.0,
                            "C2" : 6.5}
        
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
        
        self.failTable = {  "F1" : 1.5,
                            "F2" : 0.8,
                            "R1" : 1.2,
                            "R2" : 0.8,
                            "C1" : 1.5,
                            "C2" : 2.0}
        
        #Swarm-GAP work table
        self.sensors_table = np.array([[1.0, 0.0, 0.3, 0.5],
                                       [0.0, 0.0, 1.0, 0.0],
                                       [0.2, 0.0, 0.0, 1.0],
                                       [0.0, 1.0, 0.0, 0.3]])   
    def getTaskDuration(self, task_type):
        
        expectedDurations = {"Idle": 1,
                             "Att" : 1,
                             "Jam" : 1,
                             "Esc" : 1,
                             "Rec" : 1,
                             "Com" : 1,
                             }
        
        return expectedDurations[task_type]