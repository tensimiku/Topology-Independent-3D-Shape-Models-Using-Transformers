import torch
import net.networks as networks
import openmesh as om
import os


if __name__ == "__main__":
    net = networks.shapetransformernet(0, 0, 0., True)
    net.train(501, 8)


