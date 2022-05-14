

class HP:

    #encoder
    encoder_layer = 6
    encoder_dim = 128
    encoder_drop_prob = 0.1
    grapheme_size = 30
    encoder_max_input = 30

    #MHA
    nhead = 4


    encoder_feed_forward_dim = 1024
    decoder_feed_forward_dim = 1024
    feed_forward_drop_prob = 0.3

    #decoder
    decoder_layer = 6
    decoder_dim = encoder_dim
    decoder_drop_prob = 0.1
    phoneme_size = 30
    MAX_DECODE_STEP = 50

    #train










