import tensorflow as tf

class Model(object):
    
    def __init__(self):
        pass

    @staticmethod
    def start_new_session(sess):
        saver = tf.train.Saver()
        
        sess.run(tf.global_variables_initializer())
        print('started a new session')

        return saver

    @staticmethod
    def continue_previous_session(sess, ckpt_file):
        saver = tf.train.Saver()  # create a saver

        with open(ckpt_file) as file:  # read checkpoint file
            line = file.readline()  # read the first line, which contains the file name of the latest checkpoint
            ckpt = line.split('"')[1]
            model_name = ckpt.split('_')[1]

        saver.restore(sess, 'saver/'+ckpt)
        print('restored from checkpoint ' + ckpt)

        return saver, model_name
    
    @staticmethod
    def get_particular_session(sess, model_name):
        saver = tf.train.Saver()
        saver.restore(sess, 'saver/' + model_name)
        print('restored model: ' + model_name)
        
        return saver, model_name

def main():
    pass


if __name__ == '__main__':
    main()
