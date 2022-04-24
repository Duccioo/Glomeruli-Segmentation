import os
#import telepot
# choose a random element from a list
import random
import time
#import telepot.api
import urllib3
import requests
import urllib


#config
TOKEN="5084065094:AAHR1PJirn1rrvJPgM5pvJTCQc3qoeKSR2E"





def invio(string,path, now_step, total_step, CHAT_ID, testo_aggiuntivo):
    
    #prendo il path di dove sta il checkpoint e mi ricavo il FOLD corrispondente nome
    fold=path.split("/")
    mittente="sconosciuto"
    for part in fold:
        if part=="F1":
            mittente="Duccio"
        elif part=="F2":
            mittente="Giacomo"
        elif part=="F3":
            mittente="Matilde"    
        elif part=="F4":
            mittente="Renato"
        elif part=="F5":
            mittente="Gabriel"    
        

    if string=="TEST":

        if now_step==0:
            #hai appena iniziato
            testo="Grande "+mittente+" hai avviato la fase di TEST!"+"\n "+testo_aggiuntivo
        elif (now_step+1)/total_step==1:
            #finito
            testo="Non ci credo..."+mittente+"\n"+"HAI FINITO"
        else:
            if mittente=="Renato":
                testo="TEST \n"+"Renato, sei al "+str(round((now_step/total_step)*100,2))+"%"+"\n"
            
            elif mittente=="Matilde":
                testo="TEST \n"+"Mato ci sei quasi, hai fatto il "+str(round((now_step/total_step)*100,2))+"%"+"\n"+"Rilassati e prenditi due "
            
            elif mittente=="Gabriel":
                testo="TEST \n"+mittente+" "+"\n"+"Sei al "+str(round((now_step/total_step)*100,2))+"%"
            
            elif mittente=="Giacomo":
                testo="TEST \n"+"Ascolta "+"\n"+"Non ti manca molto ("+str(round((now_step/total_step)*100,2))+"%)"
            
            else:
                testo="TEST \n"+"Bravə "+mittente+" sei al "+str(round((now_step/total_step)*100,2))+"%"+"\n"     

        send_message(CHAT_ID,testo)
        time.sleep(10)
    
   
    if string=="TRAINING":

        if now_step==0:
            testo="(Training) Grande "+mittente+" hai avviato il training"+"\n"
        else:
            if mittente=="Renato":
                testo="(Training) Renato il Fascista, sei al "+str(round((now_step/total_step)*100,2))+"%"+"\n"+"BOIA CHI MOLLA"
            elif mittente=="Matilde":
                testo="(Training) Mato ci sei quasi, hai fatto il "+str(round((now_step/total_step)*100,2))+"%"+"\n"+"Rilassati e prenditi due "
            else:
                testo="(Training) Bravə "+mittente+" sei al "+str(round((now_step/total_step)*100,2))+"%"+"\n"
        
        send_message(CHAT_ID,testo)
        time.sleep(10)
    

def send_message(chat_id,testo):
    tot = urllib.parse.quote_plus(testo)
    url ="https://api.telegram.org/bot{}/".format(TOKEN) + "sendMessage?text={}&chat_id={}".format(tot, chat_id)
    get_url(url)


def get_url(url):
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content



