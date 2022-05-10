import net.transformer as transformer
import net.network as network


class shapetransformernet(network.net):
    def __init__(self, in_vtx, out_vtx, w_decay, trainable):
        super().__init__(f'./network/{self.__class__.__name__}{in_vtx}_{out_vtx}_{w_decay}', trainable, w_decay=w_decay)
        model = transformer.ShapeTransformer(in_vtx, out_vtx)
        model.cuda()
        self.model = model