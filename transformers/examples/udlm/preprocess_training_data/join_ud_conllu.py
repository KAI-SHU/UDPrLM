import sys
import os

def read_conll(file_name):
    print(file_name)
    with open(file_name, "r", encoding="utf-8") as fin:
        data = fin.readlines()
    conll_data = []
    sent_data = []
    for line in data:
        if len(line.strip()) == 0:
            if len(sent_data) > 0:
                conll_data.append(sent_data)
                sent_data = []
        else:
            line = line.strip()
            if not line.split('\t')[0].isdigit():
                continue
            sent_data.append(line)
    if len(sent_data) > 0:
        conll_data.append(sent_data)
    return conll_data

skip_lang = {"UD_Arabic-NYUAD", "UD_Hindi_English-HIENCS", "UD_English-ESL", "UD_French-FTB", "UD_Japanese-BCCWJ"}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Join UD CONLLU')
    parser.add_argument("--input_dir", type=str, help="UD Treebanks dictionary")
    parser.add_argument("--output_dir", type=str, help="output dictionary")
    args = parser.parse_args()

    train_file_name = 'train.conll'
    dev_file_name = 'dev.conll'
    labels_file_name = 'labels.txt'
    postags_file_name = 'postags.txt'
    dev_tiny_file_name = 'dev_tiny.conll'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) 

    train_join_file = open(os.path.join(args.output_dir, train_file_name), 'w', encoding="utf-8")
    dev_join_file = open(os.path.join(args.output_dir, dev_file_name), 'w', encoding="utf-8")
    dev_join_tiny_file = open(os.path.join(args.output_dir, dev_tiny_file_name), 'w', encoding="utf-8")
    
    labels_set = set()
    postags_set = set()

    for dir_name in os.listdir(args.input_dir):
        if dir_name in skip_lang:
            continue
        lang_dir = os.path.join(args.input_dir, dir_name)
        for file_name in os.listdir(lang_dir):
            f = file_name
            file_path = os.path.join(lang_dir, file_name)
            dev_count = 0
            if f.endswith("train.conllu"):
                conll_data = read_conll(file_path)
                for sent_data in conll_data:
                    for line in sent_data:
                        train_join_file.write(line+"\n")
                        tokens = line.split('\t')
                        postags_set.add(tokens[3])
                        labels_set.add(tokens[7])
                    train_join_file.write("\n")
            elif f.endswith("dev.conllu"):
                conll_data = read_conll(file_path)
                for sent_data in conll_data:
                    for line in sent_data:
                        dev_join_file.write(line+"\n")
                        tokens = line.split('\t')
                        postags_set.add(tokens[3])
                        labels_set.add(tokens[7])
                    dev_join_file.write("\n")
                for sent_data in conll_data:
                    dev_count += 1
                    if dev_count == 100:
                        break
                    for line in sent_data:
                        dev_join_tiny_file.write(line+"\n")
                    dev_join_tiny_file.write("\n")

    postags_list = ['_'] + list(postags_set)
    labels_list = ['_', 'APP'] + list(labels_set)

    with open(os.path.join(args.output_dir, postags_file_name), "w", encoding="utf-8") as fout:
        for item in postags_list:
            fout.write(item+"\n")

    with open(os.path.join(args.output_dir, labels_file_name), "w", encoding="utf-8") as fout:
        for item in labels_list:
            fout.write(item+"\n")
