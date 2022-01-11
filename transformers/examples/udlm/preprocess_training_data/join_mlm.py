import sys
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Join MLM DATA')
    parser.add_argument("--input_dir", type=str, help="MLM DATA dictionary")
    parser.add_argument("--output_dir", type=str, help="output dictionary")
    parser.add_argument("--suffix", type=str, help="suffix of needed files")
    args = parser.parse_args()

    train_file_name = 'mlm_train.txt'
    vali_file_name = 'mlm_dev.txt'
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) 

    train_join_file = open(os.path.join(args.output_dir, train_file_name), 'w', encoding="utf-8")
    vali_join_file = open(os.path.join(args.output_dir, vali_file_name), 'w', encoding="utf-8")

    for file_name in os.listdir(args.input_dir):
        if file_name.endswith(args.suffix):
            line_count = 0
            for line in open(os.path.join(args.input_dir, file_name), "r", encoding="utf-8"):
                line = line.strip()
                if len(line) > 0:
                    line_count += 1
                    if line_count <= 100:
                        vali_join_file.write(line+"\n")
                    else:
                        train_join_file.write(line+"\n")

    train_join_file.close()
    vali_join_file.close()
