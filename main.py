from Preprocessing.wsj import WSJ


wsj_path = 'ProcessedWSJ/wsj_cleaned.pkl'
wsj = WSJ()
wsj_dataset = wsj.load(wsj_path)

print(wsj_dataset[0])
