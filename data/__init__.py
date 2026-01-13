benchmarks = {
    'MIntRec': {
        'intent_labels': [
            'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize',
            'Care', 'Agree', 'Taunt', 'Flaunt',
            'Joke', 'Oppose',
            'Comfort', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave',
            'Prevent', 'Greet', 'Ask for help'
        ],
        'binary_maps': {
            'Complain': 'Emotion', 'Praise': 'Emotion', 'Apologise': 'Emotion', 'Thank': 'Emotion',
            'Criticize': 'Emotion',
            'Care': 'Emotion', 'Agree': 'Emotion', 'Taunt': 'Emotion', 'Flaunt': 'Emotion',
            'Joke': 'Emotion', 'Oppose': 'Emotion',
            'Inform': 'Goal', 'Advise': 'Goal', 'Arrange': 'Goal', 'Introduce': 'Goal', 'Leave': 'Goal',
            'Prevent': 'Goal', 'Greet': 'Goal', 'Ask for help': 'Goal', 'Comfort': 'Goal'
        },
        'label_len': 4,
        'binary_intent_labels': ['Emotion', 'Goal'],
        'max_seq_lengths': {
            'text': 37,
            'audio_feats': 480,
            'video_feats': 230
        },
        'feat_dims': {
            'text': 1024,
            'audio': 768,
            'video': 1024
        },
        'labels_map': {'Complain': 0,
                       'Praise': 1,
                       'Apologise': 2,
                       'Thank': 3,
                       'Criticize': 4,
                       'Care': 5,
                       'Agree': 6,
                       'Taunt': 7,
                       'Flaunt': 8,
                       'Joke': 9,
                       'Oppose': 10,
                       'Comfort': 11,
                       'Inform': 12,
                       'Advise': 13,
                       'Arrange': 14,
                       'Introduce': 15,
                       'Leave': 16,
                       'Prevent': 17,
                       'Greet': 18,
                       'Ask for help': 19},
        'labels_weight': {
            'Complain': 0.1289,
            'Inform': 0.1274,
            'Praise': 0.0952,
            'Apologise': 0.0615,
            'Thank': 0.0555,
            'Advise': 0.0547,
            'Criticize': 0.0525,
            'Arrange': 0.0495,
            'Introduce': 0.0472,
            'Care': 0.0427,
            'Comfort': 0.039,
            'Leave': 0.0382,
            'Prevent': 0.0322,
            'Taunt': 0.0285,
            'Greet': 0.027,
            'Agree': 0.0262,
            'Flaunt': 0.024,
            'Oppose': 0.0232,
            'Joke': 0.0232,
            'Ask for help': 0.0232
        }
    },

    'MIntRec2': {
        'intent_labels': [
            'Introduce', 'Inform', 'Explain', 'Greet', 'Ask for help',
            'Thank', 'Confirm', 'Agree', 'Apologise', 'Arrange', 'Complain',
            'Advise', 'Acknowledge', 'Warn', 'Taunt', 'Criticize', 'Care',
            'Invite', 'Comfort', 'Praise', 'Flaunt', 'Emphasize', 'Leave',
            'Prevent', 'Oppose', 'Plan', 'Doubt', 'Joke',
            'Asking for opinions', 'Refuse', 'UNK'
        ],
        'label_len': 4,
        'max_seq_lengths': {
            'text': 76,  # 50 最长的inputsID为76但是运行起来很慢
            'audio_feats': 400,
            'video_feats': 180
        },
        'feat_dims': {
            'text': 1024,  # //768//
            'audio': 768,
            'video': 256
        },
        'labels_weight': {'UNK': 0.003828,  # \\0.3828\\ !!0.003828!!
                          'Inform': 0.0604,
                          'Explain': 0.0485,
                          'Doubt': 0.0474,
                          'Complain': 0.0345,
                          'Oppose': 0.0328,
                          'Confirm': 0.0319,
                          'Praise': 0.0302,
                          'Advise': 0.0244,
                          'Agree': 0.0242,
                          'Greet': 0.02,
                          'Acknowledge': 0.0199,
                          'Thank': 0.0193,
                          'Introduce': 0.0192,
                          'Arrange': 0.019,
                          'Taunt': 0.0184,
                          'Apologise': 0.0177,
                          'Asking for opinions': 0.0166,
                          'Leave': 0.0164,
                          'Comfort': 0.0153,
                          'Care': 0.0147,
                          'Criticize': 0.0129,
                          'Plan': 0.0123,
                          'Ask for help': 0.0102,
                          'Joke': 0.0085,
                          'Prevent': 0.0082,
                          'Invite': 0.0076,
                          'Refuse': 0.007,
                          'Emphasize': 0.0067,
                          'Warn': 0.0063,
                          'Flaunt': 0.0063},
        'labels_map': {'Introduce': 0, 'Inform': 1, 'Explain': 2,
                       'Greet': 3, 'Ask for help': 4, 'Thank': 5,
                       'Confirm': 6, 'Agree': 7, 'Apologise': 8,
                       'Arrange': 9, 'Complain': 10, 'Advise': 11,
                       'Acknowledge': 12, 'Warn': 13, 'Taunt': 14,
                       'Criticize': 15, 'Care': 16, 'Invite': 17,
                       'Comfort': 18, 'Praise': 19, 'Flaunt': 20,
                       'Emphasize': 21, 'Leave': 22, 'Prevent': 23,
                       'Oppose': 24, 'Plan': 25, 'Doubt': 26, 'Joke': 27,
                       'Asking for opinions': 28, 'Refuse': 29, 'UNK': 30}
    },
    'MELD': {
        'intent_labels': [
            'Greeting', 'Question', 'Answer', 'Statement Opinion', 'Statement Non Opinion',
            'Apology', 'Command', 'Agreement', 'Disagreement',
            'Acknowledge', 'Backchannel', 'Others'
        ],
        'label_len': 3,
        'max_seq_lengths': {
            'text': 70,
            'audio_feats': 530,
            'video_feats': 250
        },
        'feat_dims': {
            'text': 1024,
            'audio': 768,
            'video': 1024
        },

        'label_maps': {
            'g': 'Greeting',
            'q': 'Question',
            'ans': 'Answer',
            'o': 'Statement Opinion',
            's': 'Statement Non Opinion',
            'ap': 'Apology',
            'c': 'Command',
            'ag': 'Agreement',
            'dag': 'Disagreement',
            'a': 'Acknowledge',
            'b': 'Backchannel',
            'oth': 'Others'
        },

        'labels_map': {
            'Greeting': 0,
            'Question': 1,
            'Answer': 2,
            'Statement Opinion': 3,
            'Statement Non Opinion': 4,
            'Apology': 5,
            'Command': 6,
            'Agreement': 7,
            'Disagreement': 8,
            'Acknowledge': 9,
            'Backchannel': 10,
            'Others': 11
        },

        'labels_weight': {
            'Statement Non Opinion': 0.34,
            'Statement Opinion': 0.20,
            'Question': 0.12,
            'Answer': 0.10,
            'Agreement': 0.05,
            'Greeting': 0.04,
            'Apology': 0.03,
            'Command': 0.03,
            'Backchannel': 0.03,
            'Disagreement': 0.02,
            'Acknowledge': 0.02,
            'Others': 0.02
        }
    }
}