import json
import os
import re

import bpy
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty

TYPE_2_PROP = {str: StringProperty, int: IntProperty, float: FloatProperty, bool: BoolProperty}

class ComfyUIAPIHandler:
    
    def __init__(self, workflow_folder_path):
        self.workflow_folder_path = workflow_folder_path
        self.load_API_files()
        self.parse_all_API_data()

    def load_API_files(self):
        self.all_api_data = {}  # {API File Name: API Json Data, }
        for filename in os.listdir(self.workflow_folder_path):
            if filename.endswith('.json'):
                with open(os.path.join(self.workflow_folder_path, filename), 'rb') as file:
                    self.all_api_data[filename] = json.load(file)
                    
        #print(f"self.all_api_data: {self.all_api_data}")
                    
    class NodeVarWrapper:
        def __init__(self, node_title, node_id, node_class_type, node_inputs):
            # remove the tag [Var] & [Order: int] and white space before them, also remove the white space at beginning or end of node title
            self.var_name = re.sub(r"\A[ \t\n\r\f\v]*|[ \t\n\r\f\v]*\[Var\]|[ \t\n\r\f\v]*\[Order:[\D]*[\d]+\]|[ \t\n\r\f\v]*\Z", "", node_title)
            order_match = re.search(r"\[Order:[\D]*([\d]+)\]", node_title)
            self.order = float('inf') if order_match is None else int(order_match.group(1))
            self.id = node_id
            self.class_type = node_class_type
            self.params = {} # {Parameters Name: Property Name, }
            for param_name in node_inputs:
                # remove numbers at beginning and non-alphanumeric: [0-9a-zA-Z_], plus node ID number at the end to avoid duplicate property names
                prop_name = re.sub(r"\A[0-9]*|[\W]+", "", param_name) + "_ID_" + str(node_id)
                param_value = node_inputs[param_name]
                if type(param_value) in TYPE_2_PROP:
                    self.set_prop(param_name, prop_name, param_value, TYPE_2_PROP[type(param_value)])
                    
            #print(f"Var Name: {self.var_name}; Parameters: {self.params}")
            
        def set_prop(self, param_name, prop_name, param_value, PropType):
            param_prop = PropType(name=prop_name, default=param_value)
            self.params[param_name] = prop_name
            setattr(bpy.types.Scene, prop_name, param_prop)
    
    def parse_all_API_data(self):
        # Get all variable nodes
        self.all_api_var = {}  # {API File Name: {Node Title: APIVarWrapper}, }
        for filename in self.all_api_data:
            api_var = {}
            api_data = self.all_api_data[filename]
            for node_id in api_data:
                node_data = api_data[node_id]
                node_title = node_data['_meta']['title']
                if '[Var]' in node_title:
                    api_var[node_title] = ComfyUIAPIHandler.NodeVarWrapper(node_title, node_id, node_data['class_type'], node_data['inputs'])
                    
            self.all_api_var[filename] = api_var
            
        # Sort the variable nodes
        self.all_node_var_sorted = {} # {API File Name: [Node Title]}
        for filename in self.all_api_var:
            api_var = self.all_api_var[filename]
    
            self.all_node_var_sorted[filename] = sorted(api_var.keys(), key=lambda node_title:api_var[node_title].order)
            
        #print(f"self.all_node_var_sorted: {self.all_node_var_sorted}")
            
    def sync_all_api_data(self):
        for filename in self.all_api_var:
            api_var = self.all_api_var[filename]
            api_data = self.all_api_data[filename]
            
            for node_title in api_var:
                node_var = api_var[node_title]
                node_var_params = node_var.params
                node_data_params = api_data[node_var.id]['inputs']
                
                for param_name in node_var_params:
                    node_data_params[param_name] = getattr(bpy.context.scene, node_var_params[param_name])
                    
        #print(f"self.all_api_data: {self.all_api_data}")

    def sync_api_data(self, filename, node_title, param_name):
        if filename in self.all_api_var and node_title in self.all_api_var[filename]:
            node_var = self.all_api_var[filename][node_title]
            if param_name in node_var.params:
                node_data_params = self.all_api_data[filename][node_var.id]['inputs']
                node_data_params[param_name] = getattr(bpy.context.scene, node_var.params[param_name])
                

if __name__ == "__main__":
    comfyUIAPIHandler = ComfyUIAPIHandler("C:/Users/reall/Softwares/Miniconda3/envs/AITexturing/_Projects/dreamgaussian/blender_py/")
    filename, node_title, param_name = "workflow_api.json", "CFG Value [Var]", "value"
    setattr(bpy.context.scene, comfyUIAPIHandler.all_api_var[filename][node_title].params[param_name], 3.5)
    #comfyUIAPIHandler.sync_api_data(filename, node_title, param_name)
    comfyUIAPIHandler.sync_all_api_data()