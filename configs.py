mnist_cfg1 = dict(dataset_type='MNIST',
                  img_shape=(1, 28, 28),
                  dim=32,
                  n_embedding=32,
                  batch_size=256,
                  n_epochs=20,
                  l_w_embedding=1,
                  l_w_commitment=0.25,
                  lr=2e-4,
                  n_epochs_2=50,
                  batch_size_2=256,
                  pixelcnn_n_blocks=15,
                  pixelcnn_dim=128,
                  pixelcnn_linear_dim=32,
                  vqvae_path='vqvae/model_mnist.pth',
                  gen_model_path='vqvae/gen_model_mnist.pth')

cifar10_cfg1 = dict(dataset_type='CIFAR10',
                    img_shape=(3, 32, 32),  
                    dim=64,  
                    n_embedding=64,
                    batch_size=128,
                    n_epochs=30,
                    l_w_embedding=1,
                    l_w_commitment=0.25,
                    lr=2e-4,
                    n_epochs_2=100,
                    batch_size_2=128,
                    pixelcnn_n_blocks=15,
                    pixelcnn_dim=256,
                    pixelcnn_linear_dim=64,
                    vqvae_path='vqvae/model_cifar10.pth',
                    gen_model_path='vqvae/gen_model_cifar10.pth')

cfgs = [mnist_cfg1, cifar10_cfg1]

def get_cfg(id: int):
    return cfgs[id]