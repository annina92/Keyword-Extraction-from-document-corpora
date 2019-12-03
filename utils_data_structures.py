import requests
import json

def get_authors_research_groups_info():
    r = requests.get("https://www.diag.uniroma1.it/utenti_gruppi_json")
    if r.status_code == 200:

        dict_author_research_group = {}
        dict_research_group_author = {}
        list_authors = []

        response_json = json.loads(r.text)

        for i in range (len(response_json["nodes"])):
            email = response_json["nodes"][i]["node"]["E-mail"]
            research_group = response_json["nodes"][i]["node"]["gruppo_ricerca"]
            if email not in list_authors:
                list_authors.append(email)
            if email in dict_author_research_group:
                dict_author_research_group[email].append(research_group)
            if email not in dict_author_research_group:
                dict_author_research_group[email] = []
                dict_author_research_group[email].append(research_group)

            if research_group in dict_research_group_author:
                dict_research_group_author[research_group].append(email)
            if research_group not in dict_research_group_author:
                dict_research_group_author[research_group] = []
                dict_research_group_author[research_group].append(email)

    print(len(dict_research_group_author.keys()))

    f = open("./data_structures/dict_author_research_group.json", "w+")
    json_data = json.dumps(dict_author_research_group)
    f.write(json_data)
    f.close()

    f = open("./data_structures/dict_research_group_author.json", "w+")
    json_data = json.dumps(dict_research_group_author)
    f.write(json_data)
    f.close()

    f = open("./data_structures/list_authors.json", "w+")
    json_data = json.dumps(list_authors)
    f.write(json_data)
    f.close()


def check_correct_papers():
    with open("./data_structures/dict_id_abstract.json", 'r+') as myfile:
        data=myfile.read()
    dict_id_abstract = json.loads(data)    
    print (len(dict_id_abstract.keys()))

def get_default_keywords():
    list_keywords = []
    r = requests.get("https://www.diag.uniroma1.it/keywords_json")
    if r.status_code == 200:
        response_json = json.loads(r.text)

        for i in range (len(response_json["nodes"])):
            list_keywords.append(response_json["nodes"][i]["node"]["name"])

    f = open("./data_structures/list_default_keywords.json", "w+")
    json_data = json.dumps(list_keywords)
    f.write(json_data)
    f.close()

def convert_surnames_email():
    with open("./data_structures/dict_authors_abstracts.json", 'r+') as myfile:
        data=myfile.read()
    authors_abstracts_dict = json.loads(data) 

    with open("./data_structures/dict_author_research_group.json", 'r+') as myfile:
        data=myfile.read()
    dict_author_research_group = json.loads(data) 

    list_authors = authors_abstracts_dict.keys()

    print(list_authors)

    dict_final_authors_research_group = {}

    for author in list_authors:
        email = author.split("@")
        name = email[0].split(".")

        if name[0]+"@diag.uniroma1.it" in dict_author_research_group:
            research = dict_author_research_group[name[0]+"@diag.uniroma1.it"]
            dict_final_authors_research_group[author] = research

        if name[1]+"@diag.uniroma1.it" in dict_author_research_group:
            research = dict_author_research_group[name[1]+"@diag.uniroma1.it"]
            dict_final_authors_research_group[author] = research

    #save dict_final_authors_research_group
    f = open("./data_structures/dict_final_authors_research_group.json", "w+")
    json_data = json.dumps(dict_final_authors_research_group)
    f.write(json_data)
    f.close()  

def check_conversion_surnames():

    with open("./data_structures/dict_final_authors_research_group.json", 'r+') as myfile:
        data=myfile.read()
    dict_final_authors_research_group = json.loads(data) 

    authors = [
    "a.deluca@uniroma1.it",
    "adriano.fazzone@uniroma1.it",
    "alberto.desantis@uniroma1.it",
    "alberto.marchetti-spaccamela@uniroma1.it",
    "alberto.nastasi@uniroma1.it",
    "alessandro.avenali@uniroma1.it",
    "alessandro.digiorgio@uniroma1.it",
    "andrea.marrella@uniroma1.it",
    "andrea.vitaletti@uniroma1.it",
    "antonio.pietrabissa@uniroma1.it",
    "antonio.sassano@uniroma1.it",
    "aris.anagnostopoulos@uniroma1.it",
    "bruno.ciciani@uniroma1.it",
    "camil.demetrescu@uniroma1.it",
    "chris.schwiegelshohn@uniroma1.it",
    "cinzia.daraio@uniroma1.it",
    "claudia.califano@uniroma1.it",
    "daniela.iacoviello@uniroma1.it",
    "daniele.nardi@uniroma1.it",
    "domenico.lembo@uniroma1.it",
    "fabio.nonino@uniroma1.it",
    "fabio.patrizi@uniroma1.it",
    "fabrizio.damore@uniroma1.it",
    "febo.cincotti@uniroma1.it",
    "fiora.pirri@uniroma1.it",
    "francesco.dellipriscoli@uniroma1.it",
    "francesco.leotta@uniroma1.it",
    "francisco.facchinei@uniroma1.it",
    "giorgio.grisetti@uniroma1.it",
    "giorgio.matteucci@uniroma1.it",
    "giuseppe.catalano@uniroma1.it",
    "giuseppe.degiacomo@uniroma1.it",
    "giuseppe.oriolo@uniroma1.it",
    "giuseppe.santucci@uniroma1.it",
    "ioannis.chatzigiannakis@uniroma1.it",
    "irene.amerini@uniroma1.it",
    "jlenia.toppi@uniroma1.it",
    "laura.astolfi@uniroma1.it",
    "laura.palagi@uniroma1.it",
    "leonardo.lanari@uniroma1.it",
    "leonardo.querzoni@uniroma1.it",
    "lorenzo.farina@uniroma1.it",
    "luca.becchetti@uniroma1.it",
    "luca.benvenuti@uniroma1.it",
    "luca.fraccascia@uniroma1.it",
    "luca.iocchi@uniroma1.it",
    "manuela.petti@uniroma1.it",
    "marco.schaerf@uniroma1.it",
    "marco.temperini@uniroma1.it",
    "marianna.desantis@uniroma1.it",
    "massimo.mecella@uniroma1.it",
    "massimo.roma@uniroma1.it",
    "maurizio.lenzerini@uniroma1.it",
    "paolo.digiamberardino@uniroma1.it",
    "paolo.liberatore@uniroma1.it",
    "pierfrancesco.reverberi@uniroma1.it",
    "renato.bruni@uniroma1.it",
    "riccardo.lazzeretti@uniroma1.it",
    "riccardo.marzano@uniroma1.it",
    "riccardo.rosati@uniroma1.it",
    "roberta.sestini@uniroma1.it",
    "roberto.baldoni@uniroma1.it",
    "roberto.beraldi@uniroma1.it",
    "roberto.capobianco@uniroma1.it",
    "rosamaria.dangelico@uniroma1.it",
    "s.lucidi@uniroma1.it",
    "salvatore.monaco@uniroma1.it",
    "silvia.bonomi@uniroma1.it",
    "simone.sagratella@uniroma1.it",
    "stefano.battilotti@uniroma1.it",
    "stefano.leonardi@uniroma1.it",
    "tiziana.catarci@uniroma1.it",
    "tiziana.dalfonso@uniroma1.it",
    "umberto.nanni@uniroma1.it",
    "valsamis.ntouskos@uniroma1.it"
    ]


    for email in dict_final_authors_research_group.keys():
        if email not in authors:
            print(email)

# ids-> research group
def get_ids_researchGroup():
    with open("./data_structures/dict_id_authors.json", 'r+') as myfile:
        data=myfile.read()
    dict_id_authors = json.loads(data) 

    with open("./data_structures/dict_final_authors_research_group.json", 'r+') as myfile:
        data=myfile.read()
    dict_final_authors_research_group = json.loads(data) 

    dict_ids_research_group = {}

    for _id in dict_id_authors.keys():
        authors = []
        authors = dict_id_authors[_id]
        research_group = []

        for author in authors:
            if author in dict_final_authors_research_group:
                for group in dict_final_authors_research_group[author]:

                    research_group.append(group)

        dict_ids_research_group[_id] = research_group

    #save
    f = open("./data_structures/dict_ids_research_group.json", "w+")
    json_data = json.dumps(dict_ids_research_group)
    f.write(json_data)
    f.close()  

def research_groups_no_duplicates():
    with open("./data_structures/dict_ids_research_group.json", 'r+') as myfile:
        data=myfile.read()
    dict_ids_research_group = json.loads(data)    
    dict_ids_rs_no_duplicates = {}

    for _id in dict_ids_research_group.keys():
        rs = dict_ids_research_group[_id]
        rs = list(dict.fromkeys(rs))
        dict_ids_rs_no_duplicates[_id] = rs

    f = open("./data_structures/dict_ids_rs_no_duplicates.json", "w+")
    json_data = json.dumps(dict_ids_rs_no_duplicates)
    f.write(json_data)
    f.close() 
    




def get_research_group_ids():
    with open("./data_structures/dict_ids_rs_no_duplicates.json", 'r+') as myfile:
        data=myfile.read()
    dict_ids_research_group = json.loads(data)    

    dict_research_group_ids = {}
    for _id in dict_ids_research_group.keys():
        rs = dict_ids_research_group[_id]
        for element in rs:
            if element in dict_research_group_ids:
                dict_research_group_ids[element].append(_id)
            if element not in dict_research_group_ids:
                dict_research_group_ids[element] = []
                dict_research_group_ids[element].append(_id)
    f = open("./data_structures/dict_research_group_ids.json", "w+")
    json_data = json.dumps(dict_research_group_ids)
    f.write(json_data)
    f.close() 
    

research_groups_no_duplicates()
get_research_group_ids()
#get_authors_research_groups_info()
#check_correct_papers()
#get_default_keywords()
#convert_surnames_email()
#get_ids_researchGroup()

