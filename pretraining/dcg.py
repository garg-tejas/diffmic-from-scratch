import torch
import torch.nn as nn
import numpy as np
import pretraining.tools as tools
import pretraining.modules as m


class DCG(nn.Module):
    def __init__(self, parameters):
        super(DCG, self).__init__()

        self.experiment_parameters = {
        "device_type": 'gpu',
        "cam_size": (16, 16),
        "K": parameters.model.num_k,
        "crop_shape": (73, 73),
        "post_processing_dim":1280,
        "num_classes":parameters.data.num_classes,
        "use_v1_global":True,
        "percent_t": 1.0,
        }
        self.cam_size = self.experiment_parameters["cam_size"]

        self.global_network = m.GlobalNetwork(self.experiment_parameters, self)
        self.global_network.add_layers()

        self.aggregation_function = m.TopTPercentAggregationFunction(self.experiment_parameters, self)
        self.retrieve_roi_crops = m.RetrieveROIModule(self.experiment_parameters, self)

        self.local_network = m.LocalNetwork(self.experiment_parameters, self)
        self.local_network.add_layers()

        self.attention_module = m.AttentionModule(self.experiment_parameters, self)
        self.attention_module.add_layers()
        self.fusion_dnn = nn.Linear(self.experiment_parameters["post_processing_dim"]+1280, self.experiment_parameters["num_classes"], bias=False)



    def _convert_crop_position(self, crops_x_small, cam_size, x_original):
        h, w = cam_size
        _, _, H, W = x_original.size()

        top_k_prop_x = crops_x_small[:, :, 0] / h
        top_k_prop_y = crops_x_small[:, :, 1] / w
        assert np.max(top_k_prop_x) <= 1.0, "top_k_prop_x >= 1.0"
        assert np.min(top_k_prop_x) >= 0.0, "top_k_prop_x <= 0.0"
        assert np.max(top_k_prop_y) <= 1.0, "top_k_prop_y >= 1.0"
        assert np.min(top_k_prop_y) >= 0.0, "top_k_prop_y <= 0.0"
        top_k_interpolate_x = np.expand_dims(np.around(top_k_prop_x * H), -1)
        top_k_interpolate_y = np.expand_dims(np.around(top_k_prop_y * W), -1)
        top_k_interpolate_2d = np.concatenate([top_k_interpolate_x, top_k_interpolate_y], axis=-1)
        return top_k_interpolate_2d

    def _retrieve_crop(self, x_original_pytorch, crop_positions, crop_method):
        batch_size, num_crops, _ = crop_positions.shape
        crop_h, crop_w = self.experiment_parameters["crop_shape"]

        output = torch.ones((batch_size, num_crops, crop_h, crop_w))
        if self.experiment_parameters["device_type"] == "gpu":
            output = output.to(x_original_pytorch.device)
        for i in range(batch_size):
            for j in range(num_crops):
                tools.crop_pytorch(x_original_pytorch[i, 0, :, :],
                                                    self.experiment_parameters["crop_shape"],
                                                    crop_positions[i,j,:],
                                                    output[i,j,:,:],
                                                    method=crop_method)
        return output



    def forward(self, x_original):
        h_g, self.saliency_map = self.global_network.forward(x_original)
        self.y_global = self.aggregation_function.forward(self.saliency_map)

        small_x_locations = self.retrieve_roi_crops.forward(x_original, self.cam_size, self.saliency_map)
        self.patch_locations = self._convert_crop_position(small_x_locations, self.cam_size, x_original)

        crops_variable = self._retrieve_crop(x_original, self.patch_locations, self.retrieve_roi_crops.crop_method)
        self.patches = crops_variable.data.cpu().numpy()
        patches = crops_variable.clone()
        
        batch_size, num_crops, I, J = crops_variable.size()
        crops_variable = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1)
        h_crops = self.local_network.forward(crops_variable).view(batch_size, num_crops, -1)
        z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)
        self.y_fusion = 0.5* (self.y_global+self.y_local)

        return self.y_fusion, self.y_global, self.y_local, patches, self.patch_attns, self.saliency_map
    