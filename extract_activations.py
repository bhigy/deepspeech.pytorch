import argparse
import json
import numpy as np
import os
import pickle
import torch
from tqdm import tqdm

from data.data_loader import SpectrogramDataset, AudioDataLoader
from ipa import arpa2ipa
from utils import load_model


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_metadata(dataset, alignments_path, half=False):
    metadata = {'audio_id': [], 'ipa': [], 'text': [], 'audio': []}
    alignments = load_alignments(alignments_path)
    alignments = [alignments[os.path.basename(id[0])] for id in dataset.ids]

    for (example, ali) in zip(dataset.ids, alignments):
        audio_path, trn_path = example[0], example[1]
        ex_id = os.path.basename(audio_path)
        metadata['audio_id'].append(ex_id)
        metadata['ipa'].append(align2ipa(ali))
        spect = dataset.parse_audio(audio_path).T
        if half:
            spect = spect.half()
        metadata['audio'].append(spect.cpu().numpy())
        with open(trn_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        metadata['text'].append(transcript)
    return metadata, alignments


def extract_activations(loader, device, model, half=False):
    model.eval()
    activations = {}
    for i, (data) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        if half:
            inputs = inputs.half()

        ac, output_sizes = model.introspect(inputs, input_sizes)

        for k in ac:
            if k not in activations:
                activations[k] = []
            for i in range(ac[k].shape[0]):
                # Need to invert the order as collate_fn use reverse order
                idx = - (i + 1)
                activations[k].append(ac[k][idx, :output_sizes[idx], :])
    return activations


def load_alignments(alignments_path):
    """Load alignment data."""
    # Reading output of forced-alignment
    alignments = {}
    for line in open(alignments_path):
        item = json.loads(line)
        if np.all([word.get('start', False) for word in item['words']]):
            audio_id = os.path.basename(item['audiopath'])
            item['audio_id'] = audio_id
            alignments[audio_id] = item
    return alignments


def align2ipa(alignment):
    """Extract IPA transcription from alignment information for a sentence."""
    result = []
    for word in alignment['words']:
        for phoneme in word['phones']:
            result.append(arpa2ipa(phoneme['phone'].split('_')[0], '_'))
    return ''.join(result)


def frames(utt, rep, index):
    """Return pair sequence of (phoneme label, frame), given an alignment
    object `utt`, a representation array `rep`, and indexing function `index`.
    """
    labels = []
    features = []
    for phoneme in phones(utt):
        phone, start, end = phoneme
        assert index(start) < index(end) + 1, \
            "Something funny: {} {} {} {}".format(start, end, index(start),
                                                  index(end))
        for j in range(index(start), index(end) + 1):
            labels.append(phone)
            features.append(rep[j])
    return features, labels


def phones(utt):
    """Return sequence of phoneme labels associated with start and end time
    corresponding to the alignment JSON object `utt`."""
    for word in utt['words']:
        pos = word['start']
        for phone in word['phones']:
            start = pos
            end = pos + phone['duration']
            pos = end
            label = phone['phone'].split('_')[0]
            if label != 'oov':
                yield (label, int(start*1000), int(end*1000))


def label_input(metadata, alignments):
    labels = []
    features = []
    for (audio, ali) in zip(metadata['audio'], alignments):
        feats, lbls = frames(ali, audio, index=lambda ms: ms // 10)
        if feats:
            labels.append(lbls)
            features.append(feats)
    return {'features': np.concatenate(features),
            'labels': np.concatenate(labels)}


def label_activations(activations, alignments):
    results = {}
    for k, v in activations.items():
        labels = []
        features = []
        for (ac, ali) in zip(v, alignments):
            feats, lbls = frames(ali, ac, index=lambda ms: ms // 20)
            if feats:
                labels.append(lbls)
                features.append(feats)
        results[k] = {'features': np.concatenate(features),
                      'labels': np.concatenate(labels)}
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracting dataset metadata')
    parser.add_argument('--cuda', action="store_true", help='Use cuda')
    parser.add_argument('--half', default=True, type=str2bool,
                        help='Use half precision. This is recommended when \
                        using mixed-precision at training time')
    parser.add_argument('--model-path-trained',
                        default='librispeech/librispeech_pretrained_v2.pth',
                        help='Path to model file created by training')
    parser.add_argument('--model-path-random',
                        default='librispeech/deepspeech_0.pth',
                        help='Path to untrained model file')
    parser.add_argument('--alignments-path',
                        default='librispeech/librispeech_val.fa.json',
                        help='Path to alignments')
    parser.add_argument('--manifest', metavar='DIR',
                        default='librispeech/libri_val_manifest_valid.csv',
                        help='path to manifest csv')
    parser.add_argument('--batch-size', default=20, type=int,
                        help='Batch size for testing')
    parser.add_argument('--num-workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--save-folder', default='librispeech',
                        help="Saves output of model to this folder")

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, args.model_path_trained, args.half)

    dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                 manifest_filepath=args.manifest,
                                 labels=model.labels, normalize=True)
    loader = AudioDataLoader(dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers)

    # Extract metadata
    metadata, alignments = extract_metadata(
        dataset=dataset,
        alignments_path=args.alignments_path,
        half=args.half)
    with open(os.path.join(args.save_folder, 'global_input.pkl'), 'wb') as f:
        pickle.dump(metadata, f, protocol=4)
    labeled_inp = label_input(metadata, alignments)
    with open(os.path.join(args.save_folder, 'local_input.pkl'), 'wb') as f:
        pickle.dump(labeled_inp, f, protocol=4)

    # Extracting activations of trained model
    activations = extract_activations(loader=loader,
                                      device=device,
                                      model=model,
                                      half=args.half)
    with open(os.path.join(args.save_folder, 'global_trained.pkl'), 'wb') as f:
        pickle.dump(activations, f, protocol=4)
    labeled_ac = label_activations(activations, alignments)
    with open(os.path.join(args.save_folder, 'local_trained.pkl'), 'wb') as f:
        pickle.dump(labeled_ac, f, protocol=4)

    # Extracting activations of random model
    model = load_model(device, args.model_path_random, args.half)
    activations = extract_activations(loader=loader,
                                      device=device,
                                      model=model,
                                      half=args.half)
    with open(os.path.join(args.save_folder, 'global_random.pkl'), 'wb') as f:
        pickle.dump(activations, f, protocol=4)
    labeled_ac = label_activations(activations, alignments)
    with open(os.path.join(args.save_folder, 'local_random.pkl'), 'wb') as f:
        pickle.dump(labeled_ac, f, protocol=4)
