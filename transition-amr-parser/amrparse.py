from transition_amr_parser.stack_transformer_amr_parser import AMRParser
import json
import torch

def parse_rams(parser, split):
    data = []
    with open('../data/rams/transfer-{}.jsonlines'.format(split)) as f:
        for line in f:
            data.append(json.loads(line))

    all_sentences = []
    for d in data:
        sentences = d['sentences']
        all_sentences.extend(sentences)
        
    with open('amr-rams-{}.txt'.format(split), 'w') as f:
        amr_list = parser.parse_sentences(all_sentences)
        for res in amr_list:
            f.write(res+'\n\n')
    torch.save(amr_list, 'amr-rams-{}.pkl'.format(split))

def parse_wikievents(parser, split):
    data = []
    with open('../data/wikievents/transfer-{}.jsonl'.format(split)) as f:
        for line in f:
            data.append(json.loads(line))

    all_sentences = []
    for d in data:
        sentences = d['sentences']
        all_sentences.extend(sentences)
        
    with open('amr-wikievent-{}.txt'.format(split), 'w') as f:
        amr_list = parser.parse_sentences(all_sentences)
        for res in amr_list:
            f.write(res+'\n\n')
    torch.save(amr_list, 'amr-wikievent-{}.pkl'.format(split))

if __name__ == '__main__':
    parser = AMRParser.from_checkpoint('../amr_general/checkpoint_best.pt')
    for split in ['train', 'dev', 'test']:
        parse_rams(parser, split)
    for split in ['train', 'dev', 'test']:
        parse_wikievents(parser, split)