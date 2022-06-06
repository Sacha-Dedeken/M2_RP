## useful code to define my main model of GAN used during this project and execute the training protocol

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from scipy.stats import chisquare

def my_GAN(source_dataset, target_dataset, n_epochs, hists_to_transf=None, gantype=0, fplot=10, rndmseed=1857, gan=None):
    """
        Unique function for the creation, training and prediction of the GAN.
    """
    tf.random.set_seed(1857)
    
    nbins = source_dataset.shape[1]
    
    if hists_to_transf == None:
        hists_to_transf = source_dataset

    a = tf.random.shuffle([1,2,3,4,5,6])
    hists_T = np.zeros(hists_to_transf.shape)
    if gan==None:
        tf.random.set_seed(1857)
        generator = keras.models.Sequential([
            keras.layers.Reshape((nbins, 1), input_shape=(nbins,)),
            keras.layers.Dense(64, activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rndmseed)),
            keras.layers.Dense(64, activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rndmseed)),
            keras.layers.Dense(64, activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rndmseed)),
            keras.layers.Conv1D(64, 3, padding="same", activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rndmseed)),
            keras.layers.Conv1D(1, 3, padding="same", activation=lambda x: keras.activations.softmax(x, axis=1), kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rndmseed)),
            keras.layers.Reshape((nbins,))
        ])

        discriminator = keras.models.Sequential([
            keras.layers.Reshape((nbins, 1), input_shape=(nbins,)),
            keras.layers.Conv1D(16, 3, padding='same',dilation_rate=2, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rndmseed)),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(32, 3, padding='same',dilation_rate=2, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rndmseed)),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.MaxPooling1D(2),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rndmseed)),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Dense(1, activation="sigmoid", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rndmseed))
        ])

        gan = keras.models.Sequential([generator, discriminator])
    print(generator.summary(), discriminator.summary())

    optimizer = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
    discriminator.trainable = False
    gan.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-4))

    batch_size = 32
    tf.random.set_seed(1857)
    target_hists = tf.random.shuffle(target_dataset)
    target_hists = tf.data.Dataset.from_tensor_slices(target_hists)
    target_hists = target_hists.batch(batch_size, drop_remainder=True).prefetch(1)
    b = tf.random.shuffle([1,2,3,4,5,6])
    
    source_hists = source_dataset[:,:,np.newaxis]
    
    history = {"dis":[], "gen":[]}
    for epoch in range(n_epochs):
        for X_batch in target_hists:
            X_batch = tf.cast(X_batch, tf.float32)

            ## Train the discriminator
            # Generator generates histograms from batch_size histograms of source_dataset 
            idx = np.random.choice(len(source_hists), batch_size, replace=False)
            generated_hists = generator(source_hists[idx])

            X_fake_and_real = tf.concat([generated_hists, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            loss_dis = discriminator.train_on_batch(X_fake_and_real, y1)

            ## Train the generator
            idx = np.random.choice(len(source_hists), batch_size, replace=False)
            input_hists = source_hists[idx]
            
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            loss_gen = gan.train_on_batch(input_hists, y2)
            
            history["dis"].append(loss_dis)
            history["gen"].append(loss_gen)
        
        if epoch % fplot == 0:
            i = np.random.randint(len(source_hists))
            plt.figure()
            plt.plot(generator.predict(np.r_[[source_hists[i]]])[0])  # histogram produced by the generator from the source histogram
            plt.plot(source_hists[i])  # source histogram
            plt.title("epoch:"+str(epoch))
            plt.show()
        print(f"{epoch:>3}/{n_epochs}  Dis. loss: {loss_dis:.4f}  Gen. loss: {loss_gen:.4f}.")
    
    for i in range(hists_T.shape[0]):
        hists_T[i] = generator.predict(np.r_[[hists_to_transf[i]]])[0]
            
    return gan, hists_T, history, a, b

def bidevice(gen):
    """
    Plot differents slices histograms for bidevice patients with the transformed histogram produced by the GAN.
    """
    
    # np.unique(hist_Skyra[hist_Skyra[:,1] == 295269][:,2]), np.unique(hist_Vida[hist_Vida[:,1] == 295269][:,2])
    # set(np.unique(hist_Skyra[:,1])).intersection(set(np.unique(hist_Vida[:,1])))
    
    exs = [(295269, 8, 7, 15), (295269, 8, 7, 26), (286860, 1, 2, 15), (286860, 1, 2, 25)] # (IPP, Skyra, Vida)
    
    fig, axs = plt.subplots(1,4, sharex=True,sharey=True, figsize=(17,5))
    
    for i, ax in enumerate(axs.flat):
        transfo = gen.predict(np.r_[[hist_Skyra[(hist_Skyra[:,1] == exs[i][0]) & (hist_Skyra[:,2] == exs[i][1]) & (hist_Skyra[:,3] == exs[i][3])][0,4:]]])[0]
        ax.plot(list(range(32)), hist_Skyra[(hist_Skyra[:,1] == exs[i][0]) & (hist_Skyra[:,2] == exs[i][1]) 
                           & (hist_Skyra[:,3] == exs[i][3])][0,4:], label="before")
        ax.plot(list(range(32)), transfo, label="after")
        ax.plot(list(range(32)), hist_Vida[(hist_Vida[:,1] == exs[i][0]) & (hist_Vida[:,2] == exs[i][2]) 
                          & (hist_Vida[:,3] == exs[i][3])][0,4:], label="target")
#         ax.plot(list(range(32)), hist_Vida[(hist_Vida[:,1] == exs[i][0]) & (hist_Vida[:,2] == exs[i][2]) 
#                           & (hist_Vida[:,3] == exs[i][3]-1)][0,4:], label="Vida", c="bisque")
#         ax.plot(list(range(32)), hist_Vida[(hist_Vida[:,1] == exs[i][0]) & (hist_Vida[:,2] == exs[i][2]) 
#                           & (hist_Vida[:,3] == exs[i][3]+1)][0,4:], label="Vida", c="bisque")
        ax.text(0, 0.32, "IPP="+str(exs[i][0])+"\nslice="+str(exs[i][3]), fontsize="large", ha="left", va="top")

    ax.legend()

##################################################################################################################

def create_GAN(nbins=32, gantype=0, rndmseed=1857):
    """
        Function that return a GAN model. 
    """
    
    tf.random.set_seed(rndmseed)
    
    if gantype==0:
        generator = keras.models.Sequential([
            keras.layers.Reshape((nbins, 1), input_shape=(nbins,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Conv1D(64, 3, padding="same", activation="relu"),
            keras.layers.Conv1D(1, 3, padding="same", activation=lambda x: keras.activations.softmax(x, axis=1)),
            keras.layers.Reshape((nbins,))
        ])

        discriminator = keras.models.Sequential([
            keras.layers.Reshape((nbins, 1), input_shape=(nbins,)),
            keras.layers.Conv1D(16, 3, padding='same',dilation_rate=2),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(32, 3, padding='same',dilation_rate=2),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.MaxPooling1D(2),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Dense(1, activation="sigmoid")
        ])

        gan = keras.models.Sequential([generator, discriminator])
    print(generator.summary(), discriminator.summary())

    optimizer = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
    discriminator.trainable = False
    gan.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-4))
    
    return gan, generator, discriminator

def train_GAN(gan, generator, discriminator, source_train, source_test, target, n_epochs=600, rndmseed=1857):
    """
        Function that train a GAN model with inputs in n_epochs.
    """
    
    print("START")
    
    np.random.seed(rndmseed)
    tf.random.set_seed(rndmseed)
    
    batch_size = 32
    tf.random.set_seed(1857)
    target_hists = tf.random.shuffle(target)
    target_hists = tf.data.Dataset.from_tensor_slices(target_hists)
    target_hists = target_hists.batch(batch_size, drop_remainder=True).prefetch(1)
    
    source_hists = source_train[:,:,np.newaxis]
    
    history = {"dis":[], "gen":[], "score_train_m":[], "score_test_m":[], "score_train_v":[], "score_test_v":[]}
    
    print("TRAINING")
    
    for epoch in range(n_epochs):
        for X_batch in target_hists:
            X_batch = tf.cast(X_batch, tf.float32)

            ## Train the discriminator
            # Generator generates histograms from batch_size histograms of source_dataset 
            idx = np.random.choice(len(source_hists), batch_size, replace=False)
            generated_hists = generator(source_hists[idx])

            X_fake_and_real = tf.concat([generated_hists, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            loss_dis = discriminator.train_on_batch(X_fake_and_real, y1)

            ## Train the generator
            idx = np.random.choice(len(source_hists), batch_size, replace=False)
            input_hists = source_hists[idx]
            
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            loss_gen = gan.train_on_batch(input_hists, y2)
            
            history["dis"].append(loss_dis)
            history["gen"].append(loss_gen)
            
            # Test the discriminator and the generator
            
        if epoch%50 == 49:
            print("test")
            hists_T_test = transformation(generator, source_test)
            hists_T_train = transformation(generator, source_train)
            print("fin transformation")
            score_train_m, score_train_v = evaluate_GAN(source_train, hists_T_train, target, base=True)
            score_test_m, score_test_v = evaluate_GAN(source_test, hists_T_test, target, base=True)
            print("fin evaluate")
            history["score_train_m"].append(score_train_m)
            history["score_test_m"].append(score_test_m)
            history["score_train_v"].append(score_train_v)
            history["score_test_v"].append(score_test_v)

            print(f"***** {epoch:>3}/{n_epochs}  Train score: {score_train_m:.0f}% (m) {score_train_v:.0f}% (v) | Test score: {score_test_m:.0f}% (m) {score_test_v:.0f}% (v)")
        print(f"{epoch:>3}/{n_epochs}  Dis. loss: {loss_dis:.3f}  Gen. loss: {loss_gen:.3f}.")
    
    return gan, generator, discriminator, history

def transformation(generator, hists_to_transf):
    """
        Function that use the GAN generator on a dataset.
    """
    hists_T = np.zeros(hists_to_transf.shape)
    for i in range(hists_T.shape[0]):
        hists_T[i] = generator.predict(np.r_[[hists_to_transf[i]]])[0]
    
    return hists_T

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def loss_plot(history_loss, n_epochs):
    """
        Function that plot the loss_history of the GAN generator and discriminator and their moving average.
    """

    t = np.linspace(0, n_epochs, len(history_loss["dis"]))
    plt.plot(t, history_loss["dis"], alpha=0.2, c="orange")
    plt.plot(t, history_loss["gen"], alpha=0.2, c="blue")

    ma_d = moving_average(history_loss["dis"], int(len(history_loss["dis"])/70))
    ma_g = moving_average(history_loss["gen"], int(len(history_loss["gen"])/70))
    tbis = np.linspace(0, n_epochs, len(ma_d))
    plt.plot(tbis, ma_g, c="blue", label="generator")
    plt.plot(tbis, ma_d, c="orange", label="discriminator")
    plt.title("(averaged) loss function of each part of the GAN by epoch")
    plt.legend()

    print("slope after the first third:", (ma_g[-1] - ma_g[-int(2*len(ma_g)/3)])/int(2*len(ma_g)/3) )
    
    
def evaluate_GAN(before, after, target, base=False):
    """
        Function that print and plot the GAN evaluation on the first and second moments convergence.
    """
    mean_before, mean_after, mean_target = np.mean(before, axis=0), np.mean(after, axis=0), np.mean(target, axis=0)
    var_before, var_after, var_target = np.var(before, axis=0), np.var(after, axis=0), np.var(target, axis=0)
    
    if base:
        return 100*(chisquare(mean_before, mean_target)[0] - chisquare(mean_after, mean_target)[0])/chisquare(mean_before, mean_target)[0], 100*(chisquare(var_before, var_target)[0] - chisquare(var_after,var_target)[0])/chisquare(var_before,var_target)[0]
    
    fig, axs = plt.subplots(1, 2, figsize=(15,10))

    axs[0].plot(mean_before, label="before")
    axs[0].plot(mean_after, label="after")
    axs[0].plot(mean_target, label="target")
    axs[0].title.set_text("mean") 
    axs[0].legend()
    
    axs[1].plot(var_before, label="before")
    axs[1].plot(var_after, label="after")
    axs[1].plot(var_target, label="target")
    axs[1].title.set_text("variance")
    axs[1].legend()
    
    print("Chi square distance with target before/after : {:.4f} -> {:.4f} | Score : {:.0f}%".format(chisquare(mean_before, mean_target)[0], chisquare(mean_after, mean_target)[0], 100*(chisquare(mean_before, mean_target)[0] - chisquare(mean_after, mean_target)[0])/chisquare(mean_before, mean_target)[0]))
    
    print("Variance distance with target before/after : {:.4f} -> {:.4f} | Score : {:.0f}%".format(chisquare(var_before, var_target)[0], chisquare(var_after,var_target)[0], 100*(chisquare(var_before, var_target)[0] - chisquare(var_after,var_target)[0])/chisquare(var_before,var_target)[0]))






