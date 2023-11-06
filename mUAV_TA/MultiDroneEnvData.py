import numpy as np


class SceneData:
    def __init__(self):
        
        # Tamanho da tela        
        self.GameArea = [1200,700] #each pixel represent 0.5nm
        self.ContactLine = 550
        self.Bases = [np.array([400,680])]
        
        self.UavTypes = [ "R1", "R2", "E1" , "F1", "F2", "T1" , "T2"]
                        #  0     1      2      3     4     5      6
        
        self.TaskTypes =                    ["Hold", "Rec" , "Att" , "Def" ,  "Int" ,  "Det" ]
        self.UavCapTable ={ "R1" : np.array([ 0.1  ,  1.0  ,  0.0  ,  0.2  ,   0.0  ,   0.0  ]),
                            "R2" : np.array([ 0.1  ,  0.6  ,  0.0  ,  0.1  ,   0.0  ,   0.0  ]),
                            "E1" : np.array([ 0.1  ,  0.8  ,  0.0  ,  0.2  ,   0.0  ,   1.0  ]),
                            "F1" : np.array([ 0.1  ,  0.0  ,  0.7  ,  1.0  ,   1.0  ,   1.0  ]),
                            "F2" : np.array([ 0.1  ,  0.0  ,  1.0  ,  0.6  ,   0.8  ,   1.0  ]),                            
                            "T1" : np.array([ 0.0  ,  0.0  ,  0.2  ,  0.5  ,   1.0  ,   1.0  ]),
                            "T2" : np.array([ 0.0  ,  0.0  ,  0.2  ,  0.4  ,   0.8  ,   0.8  ])}
                
        self.UavCapTableByIdx   =  np.array([self.UavCapTable[ut] for ut in self.UavTypes])
        
        self.UavIndex =  {name : idx  for idx, name in enumerate(self.UavTypes)}
        self.TaskIndex = {name : idx  for idx, name in enumerate(self.TaskTypes)}
        
        self.maxSpeeds = {  "F1" : 20.0,
                            "F2" : 15.0,
                            "R1" : 5.0,
                            "R2" : 8.0,  
                            "E1" : 5.0,                                                      
                            "T1" : 14.0,
                            "T2" : 12.0}
        
        self.Endurances = { "F1" : 1000,
                            "F2" : 800,
                            "R1" : 2000,
                            "R2" : 2000,
                            "E1" : 3500,                            
                            "T1" : 1200.0,
                            "T2" : 1800.0}
        
        self.engage_range ={"F1" : 40.0,
                            "F2" : 30.0,
                            "R1" : 0.0,
                            "R2" : 0.0,
                            "E1" : 0.0,                            
                            "T1" : 35.0,
                            "T2" : 25.0}
        
        relayFactor = 250
        self.relayArea = {uavType : relayFactor * self.UavCapTable[uavType][4] 
                            for uavType in self.UavTypes}
        
        self.failTable = {  "F1" : 1.5,
                            "F2" : 0.8,
                            "R1" : 1.2,
                            "R2" : 0.8,
                            "E1" : 1.5,                            
                            "T1" : 1.8,
                            "T2" : 1.0}
        
        #Swarm-GAP work table
        self.sensors_table = np.array([[1.0, 0.0, 0.3, 0.5],
                                       [0.0, 0.0, 1.0, 0.0],
                                       [0.2, 0.0, 0.0, 1.0],
                                       [0.0, 1.0, 0.0, 0.3]])   
    def getTaskDuration(self, task_type):
        
        expectedDurations = {"Idle": 1,
                             "Att" : 5,
                             "Def" : 5,
                             "Det" : 1,
                             "Esc" : 1,
                             "Rec" : 10,
                             "Int" : 0,
                             "Hold": 1,
                             }
        
        return expectedDurations[task_type]