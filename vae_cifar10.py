import datasets
import vae
import train_vae



if __name__=='__main__':
    data = datasets.get_dataset()
    model = vae.VAE()
    model = train_vae.train_vae(model, data)


