import pandas as pd
import numpy as np
import os, re, json


class RuleIntel:
    def __init__(self):
        self.x=1
        
        '''
        loads the class with the necessary reference
        inputs: alarmdict(new alarm json file) and previous data frame
        
        '''
    def load_input(self,alarmdict,dataframe):
        self.alarmdict = alarmdict
        self.aldate = self.alarmdict['FIRSTOCCURRENCE']
        self.f_eve_sum = self.alarmdict['FIRSTEVENTSUMMARY']
        self.node = self.alarmdict['NODE']
        self.cls = self.alarmdict['CLASS']
        self.datediff= None
        self.BMCStatus = 'Open'
        self.dataframe = dataframe.copy()
        self.orig_df = dataframe.copy()
        self.last_alarm_count=0
        return "<<<<<<< Object Loaded and Initialized >>>>>>>"
    
    '''
    gets the last alarm count for the new alarm to be appened with the updated count. 
    function "load_input" has to be executed to set the class reference before executing
    this function
    '''
    def getlastalarmcount(self):
        if(bool(self.dataframe[(self.dataframe['FIRSTEVENTSUMMARY'] == self.f_eve_sum) 
                               & (self.dataframe['NODE'] == self.node)
                               &(self.dataframe['CLASS'] == self.cls)
                               & (self.dataframe['BMC_Status'] == self.BMCStatus)].shape[0])):
            self.last_alarm_count = self.dataframe[(self.dataframe['FIRSTEVENTSUMMARY'] == self.f_eve_sum) & (self.dataframe['NODE'] == self.node) & (self.dataframe['CLASS'] == self.cls)
                                                   & (self.dataframe['BMC_Status'] == self.BMCStatus)].sort_values('FIRSTOCCURRENCE',ascending=False)[:1]['Count'].values[0]
            self.datediff = True
                                                    
            self.last_alarm_count = self.last_alarm_count +1
        else:
            self.datediff = False
            self.last_alarm_count=1
        self.alarmdict['Count'] = self.last_alarm_count
        return(self.alarmdict)
    
    
    '''
    Appends the new alarm into the previos version of the dataframe. Load data and getlastalarmcount
    should be executed prior to this
    '''
    def appendrecord(self):
        self.orig_df = self.orig_df.append((self.alarmdict),ignore_index=True)
        self.orig_df['FIRSTOCCURRENCE'] = pd.to_datetime(self.orig_df['FIRSTOCCURRENCE'])
        return(self.orig_df)
    
    
    '''
    gets the time difference between the first and latest alarm in the series
    input: dataframe after append
    
    '''
    def gettimeinterval(self,inpdf):
        self.inpdataframe= inpdf
        if(self.datediff):
            self.dflen = len(self.inpdataframe[(self.inpdataframe['FIRSTEVENTSUMMARY'] == self.f_eve_sum) & (self.inpdataframe['NODE'] == self.node) & (self.inpdataframe['CLASS'] == self.cls)
                                                   & (self.inpdataframe['BMC_Status'] == self.BMCStatus)].sort_values('FIRSTOCCURRENCE',ascending=False))
            self.date1 = self.inpdataframe[(self.inpdataframe['FIRSTEVENTSUMMARY'] == self.f_eve_sum) & (self.inpdataframe['NODE'] == self.node) & (self.inpdataframe['CLASS'] == self.cls)
                                                   & (self.inpdataframe['BMC_Status'] == self.BMCStatus)].sort_values('FIRSTOCCURRENCE',ascending=False)[:1]['FIRSTOCCURRENCE'].values[0]
            
            self.date2=self.inpdataframe[(self.inpdataframe['FIRSTEVENTSUMMARY'] == self.f_eve_sum) & (self.inpdataframe['NODE'] == self.node) & (self.inpdataframe['CLASS'] == self.cls)
                                                   & (self.inpdataframe['BMC_Status'] == self.BMCStatus)].sort_values('FIRSTOCCURRENCE',ascending=False)[(self.dflen-1):]['FIRSTOCCURRENCE'].values[0]
           
            self.minutes = (pd.to_datetime(self.date1) - pd.to_datetime(self.date2)).total_seconds() / 60

            return self.minutes
        else:
            self.minutes = 0
            return self.minutes
    
    '''
    generates unique id for unique list of actions. Transforms the same to json 
    Inputs: rulebook data frame and sequence number
    
    '''
    def GenerateRuleID_tojson(self,rulebook,rule_seq):
        self.rule_book = rulebook
        self.rule_seq = rule_seq
        self.action_taken_list = list(self.rule_book['Action Taken'].unique())
        #idx = 1101
        for self.k in range(0,len(self.action_taken_list)):
            #print(action_taken_list[k])
            for self.i,self.value in self.rule_book[self.rule_book['Action Taken'] == self.action_taken_list[self.k]].iterrows():
                #print(idx)
                self.rule_book.loc[self.i,'Rule_Id']= (self.rule_seq)
                #df.loc[i,'timediff']= (df[df.Val == 'ticket3'].loc[i,'Date'] - df[df.Val == 'ticket3'].loc[i-1,'Date']).total_seconds() / 60
                self.x+=1
            self.rule_seq+=1
        self.dict_raise_rule = self.rule_book.groupby('Action Taken')['Rule_Id'].unique().apply(list).to_dict()
            
        json.dump(self.dict_raise_rule,open ('dict_map_rule.json','+w'))
        self.getdir= os.getcwd()
        return f"Dictionary file  >> 'dict_map_rule.json' for rules has been created in >>>>>> {self.getdir}"
    
    '''
    Maps the alarm with the rule and returns the action statement
    inputs: alarm data frame, updated dataframe , rulebook dataframe and 
    action-ruleid json file generated by GenerateRuleID_tojson
    '''
    def Map_RuleId(self,alarm,appenddf,rulebook,rule_json):
        self.alarm = alarm
        self.rulebook = rulebook
        self.rule_json = rule_json
        self.appenddf = appenddf
        
        for self.i,self.value in self.alarm.iterrows():
            self.Node = (self.alarm.loc[self.i,'NODE'])
            self.FES = (self.alarm.loc[self.i,'FIRSTEVENTSUMMARY'])
            self.CLS = (self.alarm.loc[self.i,'CLASS'])
            #print(Node,'\n',FES,'\n',CLS)
            self.df_rule_filter = self.rulebook[(self.rulebook['NODE'] == self.Node) & (self.rulebook['FIRSTEVENTSUMMARY'] == self.FES) & (self.rulebook['CLASS'] == self.CLS)]['Action Taken']
            #print(self.df_rule_filter.values)
            if(len(self.df_rule_filter.values)>0):
                self.df_rule_filter_value = self.df_rule_filter.values[0]
                self.dict_ruleid = self.rule_json[self.df_rule_filter_value]
                #print(self.dict_ruleid)
                self.dictruleid = self.dict_ruleid[0]#.astype('int')
                #print(self.dictruleid)
                self.ret_val = self.Choice(self.dictruleid)
                #print(self.ret_val)
                for self.l,self.valu in self.appenddf[(self.appenddf['Rule_Action'] == '') & (self.appenddf['BMC_Status'] == 'Open')].iterrows():
                    #print(self.l)
                    self.appenddf.loc[self.l,'Rule_Action'] = self.ret_val.split('|')[0]
                    self.appenddf.loc[self.l,'Rule_Ticket'] = [1 if(len(self.ret_val.split('|'))) == 2 else 0]
                #self.alarm.loc[self.i,'Rule_Node'] = self.Node
                #self.alarm.loc[self.i,'Rule_FES'] = self.FES
                #self.alarm.loc[self.i,'Rule_CLS'] = self.CLS
        
        return self.appenddf
    

    def Choice(self,dictruleid):
        self.dicid = dictruleid
        self.switcher = {
                1106: self.rule_1106,
                1101: self.rule_1101,
                1102: self.rule_1102,
                1103: self.rule_1103,
                1104: self.rule_1104,
                1105: self.rule_1105,
                1117: self.rule_1117,
                1107: self.rule_1107,
                1108: self.rule_1108,
                1109: self.rule_1109,
                1110: self.rule_1110,
                1111: self.rule_1111,
                1112: self.rule_1112,
                1113: self.rule_1113,
                1114: self.rule_1114,
                1115: self.rule_1115,
                1116: self.rule_1116,
                1118: self.rule_1118,
                1119: self.rule_1119,
                1120: self.rule_1120,
                1121: self.rule_1121,
                1123: self.rule_1123,
                1124: self.rule_1124,
                1125: self.rule_1125,
                1128: self.rule_1128,
                1129: self.rule_1129,
                1130: self.rule_1130,
                1131: self.rule_1131,
                1132: self.rule_1132,
                1133: self.rule_1133,
                1134: self.rule_1134,
                1135: self.rule_1135,
                1136: self.rule_1136,
                1137: self.rule_1137,
                1138: self.rule_1138,
#                 1139: self.rule_1139,
                1140: self.rule_1140,
                1141: self.rule_1141,
                1142: self.rule_1142,
                1143: self.rule_1143,
                1144: self.rule_1144,
                1145: self.rule_1145,
                1146: self.rule_1146,
                1147: self.rule_1147,
                1148: self.rule_1148,
                1150: self.rule_1150,
                1151: self.rule_1151,

  
                #3: lambda: 'two'
            }
        self.func = self.switcher.get(self.dicid, lambda:'Invalid Rule Id. Check with you system administrator!')
        return self.func()

    def rule_1101(self):
        return "Raise P4 case to Integrate L2 queue|1"


    def rule_1102(self):
        return "Raise P4 case and drop mail to storage team|1"
    

    def rule_1103(self):
        return "Raise P4 case if alert  comes with count 1. More than count 1 no need to raise case|1"
    

    def rule_1104(self):
        return "Raise P4 case to Wintel Server support|1"


    def rule_1105(self):
        return "Raise P4 case to Server Team(OS Team)|1"


    def rule_1106(self):
        return "Raise case from Netcool and drop mail to EOTS team|1"


    def rule_1107(self):
        return "Raise case and drop mail to server team|!"


    def rule_1108(self):
        return "Raise P4 case and drop mail to BMC Patrol team|!"


    def rule_1109(self):
        return "Raise case to BMC Patrol team and drop mail|1"


    def rule_1110(self):
        return "Raise case and drop mail to Database Team|1"


    def rule_1111(self):
        return "Raise case from Netcool and drop mail to Operational Intelligence team(SMIP)|1"


    def rule_1112(self):
        return "Raise case and drop mail to messaging team. Give callout|1"


    def rule_1113(self):
        return "Raise case and drop mail to app team|1"


    def rule_1114(self):
        return "Raise P4 case and drop mail to server team|1"


    def rule_1115(self):
        return "Raise case from Netcool and drop mail to Patrol team|1"


    def rule_1116(self):
        return "Raise case and drop mail to BMC Patrol team|1"


    def rule_1117(self):
        return "Raise case from Netcool and drop mail|1"
    

    def rule_1118(self):
        return "Check with Wintel L2 before raising Incidents"
    

    def rule_1119(self):
        return "Check with ITCC Unix before raising case"
    
    def rule_1120(self):
        if (int(self.minutes)>= 30):
            return "Raise Ticket for Alarm to IT Platforms because alarm is not cleared within 30 minutes|1"
        else:
            return "Alarm wait time is within BAU(30 mins). Do not raise ticket"
        
    def rule_1121(self):
        if (int(self.minutes)< 15):
            return f"Raise Ticket for Alarm as it did not get cleared within {self.minutes} mins|1"
        else:
            return "Alarm wait time is within BAU(15 mins). Do not raise ticket"
        
    def rule_1122(self):
        if (int(self.minutes)> 10):
            return f"Raise Ticket for Alarm as it did not get cleared within {self.minutes} mins|1"
        else:
            return "Alarm wait time is within BAU(5 - 10 mins). Do not raise ticket"
        
        
    def rule_1123(self):
        if (int(self.minutes)> 30):
            return "Raise Ticket for Alarm to micropayments team as it did not get cleared within 30 mins|1"
        else:
            return "Alarm wait time is within BAU(30 mins). Do not raise ticket"
        
    def rule_1124(self):
        if (int(self.minutes)< 30 & (self.last_alarm_count > 12)):#10-15 cnt
            return f"Raise Ticket for Alarm as the count has reached ({self.last_alarm_count}) within 30 mins|1"
        else:
            return "Alarm wait time is within BAU(30 mins). Do not raise ticket"
        
    def rule_1125(self):
        if (int(self.minutes)< 59 & int(self.last_alarm_count > 8) ):# multiple in 1 hr
            return f"Raise Ticket for Alarm as count has reached ({self.last_alarm_count}) within ({self.minutes})|1"
        else:
            return "Alarm wait time is within BAU(60 mins). Do not raise ticket"
        
    def rule_1126(self):
        if (int(self.minutes) > 59 & int(self.last_alarm_count > 5) ):
            return f"Give callout to CPI/CPS Support as Alarm count has reached ({self.last_alarm_count}) within ({self.minutes}) min"
        else:
            return f"Raise p4 ticket for alarm as count reached ({self.last_alarm_count}) within ({self.minutes}) min|1 "
        
    def rule_1127(self):
        if (int(self.minutes)< 30 & int(self.last_alarm_count > 12) ):#10-15 cnt
            return f"Raise ticket,Email and make call to Micropayments team as Alarm count has reached ({self.last_alarm_count}) within ({self.minutes}) min|1"
        else:
            return f"Alarm wait time is within BAU({self.minutes} min(s)). Do not raise ticket"    
        
    def rule_1128(self):
        if (int(self.minutes)< 10 & int(self.last_alarm_count > 30) ):
            return f"Email to Convergys as Alarm count has reached ({self.last_alarm_count}) within 10 min"
        else:
            return "Alarm wait time is within BAU(10 min(s)). Do not raise ticket"
        
    def rule_1129(self):
        if (int(self.minutes)< 1 & int(self.last_alarm_count > 10) ):
            return f"Call out to Convergys as Alarm count has reached ({self.last_alarm_count}) within 1 min"
        else:
            return "Alarm wait time is within BAU(1 min(s)). Do not raise ticket"
        
    def rule_1130(self):
            if (int(self.last_alarm_count >= 1)):#first cnt
                return f"Raise Ticket and dive callout for Alarm as the count has reached ({self.last_alarm_count})|1"
            else:
                return "Alarm count is within BAU(1 count).Do not raise ticket"

    def rule_1131(self):
            if (int(self.last_alarm_count >= 1)):#fst cnt
                return f"Raise Ticket and give callout to CPS Team as the count has reached ({self.last_alarm_count})|1"
            else:
                return "Alarm count is within BAU(1 count).Do not raise ticket"

    def rule_1132(self):
            if (int(self.last_alarm_count >= 1)):#Fst cnt
                return f"Raise Ticket, Give callout and Email to Micropayments team for Alarm as the count has reached ({self.last_alarm_count})|1"
            else:
                return "Alarm count is within BAU(1 count).Do not raise ticket"

    def rule_1133(self):
            if (int(self.last_alarm_count >= 1)):#fst cnt
                return f"Raise Ticket, Give callout and Email to Micropayments team for Alarm as the count has reached {self.last_alarm_count}|1"
            else:
                return "Alarm count is within BAU(1 count).Do not raise ticket"

    def rule_1134(self):
            if (int(self.last_alarm_count >= 1)):#fst cnt
                return f"Raise Ticket, Give callout and Email to CPS team for Alarm as the count has reached ({self.last_alarm_count})|1"
            else:
                return "Alarm count is within BAU(1 count).Do not raise ticket"

    def rule_1135(self):
            if (int(self.last_alarm_count >= 1)):#fst cnt
                return f"Raise Ticket and Give callout for Alarm as the count has reached ({self.last_alarm_count})|1"
            else:
                return "Alarm count is within BAU(1 count).Do not raise ticket"

    def rule_1136(self):
            if (int(self.last_alarm_count >= 1)):#fst cnt
                return f"Raise Ticket and give callout to CPS Team as the count has reached ({self.last_alarm_count})|1"
            else:
                return "Alarm count is within BAU(1 count).Do not raise ticket"

    def rule_1137(self):
            if (int(self.last_alarm_count >= 1)):#fst cnt
                return f"Raise p2 Ticket and give callout to CPS Team as the count has reached ({self.last_alarm_count})|1"
            else:
                return "Alarm count is within BAU(1 count).Do not raise ticket"

    def rule_1138(self):
            if (int(self.minutes)> 5 & int(self.last_alarm_count > 3) ):
                return f"Raise Ticket and Give callout for Alarm as count has reached ({self.last_alarm_count}) within ({self.minutes})|1"
            else:
                return f"Alarm wait time is within BAU({self.minutes} mins). Do not raise ticket"


    def rule_1140(self):
            if (int(self.minutes)> 5 & int(self.last_alarm_count >= 5) ):
                return f"Raise p3 Ticket and Give callout for Alarm as count has reached ({self.last_alarm_count}) within ({self.minutes})|1"
            else:
                return f"Alarm wait time is within BAU({self.minutes} mins). Do not raise ticket"

    def rule_1141(self):
            if (int(self.minutes)> 30):
                return f"Raise Ticket and Email for Alarm as it did not get cleared within ({self.minutes}) mins|1"
            else:
                return f"Alarm wait time is within BAU({self.minutes} min). Do not raise ticket"

    def rule_1142(self):
            if (int(self.minutes)> 5 & int(self.last_alarm_count >= 3) ):
                return f"Give callout with case reference for Alarm as count has reached ({self.last_alarm_count}) within ({self.minutes}) "
            else:
                return f"Alarm wait time is within BAU({self.minutes} mins). Do not raise ticket"

    def rule_1143(self):
            if (int(self.last_alarm_count > 12)):#10-15 cnt
                return f"Notify micropayments team as the count has reached ({self.last_alarm_count})"
            else:
                return f"Alarm count is within BAU({self.last_alarm_count}).Do not raise ticket"

    def rule_1144(self):
            if (int(self.last_alarm_count > 12)):#10-15 cnt
                return f"Give callout as the count has reached ({self.last_alarm_count})"
            else:
                return "Alarm count is within BAU(<10 count).Do not raise ticket"
            
    def rule_1145(self):
        if (int(self.last_alarm_count >= 5) ):
            return f"Give callout for Alarm as count has reached ({self.last_alarm_count})"
        else:
            return "Alarm count is within BAU(5 count).Do not raise ticket"

    def rule_1146(self):
            if (int(self.minutes)> 20):
                return f"Raise Ticket and Email for Alarm as it did not get cleared within ({self.minutes}) mins|1"
            else:
                return "Alarm wait time is within BAU(20 min). Do not raise ticket"

    def rule_1147(self):
            if (int(self.last_alarm_count >= 3)):#10-15 cnt
                return f"Give callout as the count has reached ({self.last_alarm_count})"
            else:
                return "Alarm count is within BAU(3 count).Do not raise ticket"

    def rule_1148(self):
            if (int(self.last_alarm_count > 10)):
                return f"Raise p2 ticket and Give callout as the count has reached ({self.last_alarm_count})|1"
            else:
                return f"Raise low priority ticket as the count has reached {self.last_alarm_count}"

    def rule_1149(self):
            if (int(self.minutes)> 30 & int(self.last_alarm_count >= 5) ):
                return f"Raise Ticket and Give callout for Alarm as count has reached ({self.last_alarm_count}) within ({self.minutes})|1"
            else:
                return f"Alarm wait time is within BAU({self.minutes} mins). Do not raise ticket"

    def rule_1150(self):
            if (int(self.last_alarm_count > 5)):
                return f"Raise p2 ticket and Give callout as the count has reached ({self.last_alarm_count})|1"
            else:
                return f"Raise p3 ticket as the count has reached ({self.last_alarm_count})|1"

    def rule_1151(self):
            if (int(self.minutes)> 10 & int(self.last_alarm_count > 5) ):#3-5 cnt
                return f"Give callout for Alarm as count has reached ({self.last_alarm_count}) within ({self.minutes}) "
            else:
                return f"Raise ticket and Email as count has reached ({self.last_alarm_count}) within ({self.minutes})|1"

  