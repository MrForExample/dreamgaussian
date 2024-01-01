bl_info = {
    'name': "BlenderAI43D",
    "author": "Mr. For Example",
    "version": (0, 0, 1),
    "blender": (4, 0, 0),
    "location": "3D Viewport > Sidebar > BlenderAI43D",
    "description": "Using the power of AI to create 3D assests in Blender",
    "category": "Development",
}


import bpy
import json
import os
import re
import hashlib
import enum

from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty
from bpy.types import Operator, Panel

WORKFLOW_FOLDER_PATH_PROP = {'name': "workflow_api_folder_path", 'prop': StringProperty(name="workflow_api_folder_path")}
CFG_PROP = {'name': "CFG", 'prop': FloatProperty(default=0, name="CFG")}
CHECKPOINT_PROP = {'name': "checkpoint_name", 'prop': EnumProperty(items=(("SD1.5", "SD1.5", "SD1.5"), ("SDXL", "SDXL", "SDXL")), name="checkpoint_name")}

all_props = (
    WORKFLOW_FOLDER_PATH_PROP,
    CFG_PROP,
    CHECKPOINT_PROP,
)


REF_STYLE_IMG_NAME = "ref_style_img"
REF_FACE_STYLE_IMG_NAME = "ref_face_style_img"

all_imgs_names = (
    REF_STYLE_IMG_NAME,
    REF_FACE_STYLE_IMG_NAME,
)

AI_TEXTURING_WORKFLOW_API_NAMES = (
    "0_Multi-View_Images_Generation_api.json",
    "1_Face_Detailer (Optional)_api.json",
    "2_Upscale_api.json",
    "3_VFI (Optional)_api.json",
)

SYNC_TYPES = ["Customize", "SyncToFirst", "SyncToLast"]
SYNC_TYPES_DESCRIPTION = [
    "Manually set the parameters of this node",
    "Synchronize the parameters of this node to the first node from previous stages that has same name & type as this node",
    "Synchronize the parameters of this node to the last node from previous stages that has same name & type as this node"
]

COMFYUI_INPUT_FOLDER_RELATIVE_PATH = "ComfyUI/input/"
STYLE_IMG_FOLDER_NAME = "style/"


TYPE_2_PROP = {str: StringProperty, int: IntProperty, float: FloatProperty, bool: BoolProperty}

def sync_child_nodes_properties(self, context, node_var, param_name):
    if hasattr(context.scene, node_var.params[param_name]):
        target_prop = getattr(context.scene, node_var.params[param_name])
        child_params = node_var.child_params[param_name]
        for child_prop_name in child_params:
            setattr(context.scene, child_prop_name, target_prop)
    
def set_parent_node_to_sync(self, context, child_node_var):
    # remove parameters from current node to sync
    if child_node_var.now_nodes_to_sync_index > 0:
        child_params = child_node_var.nodes_to_sync[child_node_var.now_nodes_to_sync_index].child_params
        for param_name in child_node_var.params:
            child_prop_name = child_node_var.params[param_name]
            if child_prop_name in child_params[param_name]:
                child_params[param_name].remove(child_prop_name)
    
    # add child_node_var to new node to sync
    child_node_var.now_nodes_to_sync_index = SYNC_TYPES.index(getattr(context.scene, child_node_var.node_sync_enum_prop_name))
    node_var = child_node_var.nodes_to_sync[child_node_var.now_nodes_to_sync_index]
    if child_node_var.node_type == ComfyUIAPIHandler.NodeType.IMGS:
        child_node_var.params[child_node_var.image_param_name] = node_var.img_prop_name
    elif child_node_var.now_nodes_to_sync_index > 0:
        child_params = node_var.child_params
        for param_name in child_node_var.params:
            child_prop_name = child_node_var.params[param_name]
            child_params[param_name].append(child_prop_name)
            if hasattr(context.scene, node_var.params[param_name]):
                target_prop = getattr(context.scene, node_var.params[param_name])
                setattr(context.scene, child_prop_name, target_prop)

class ComfyUIAPIHandler:
    
    def __init__(self, comfyUI_root_path, workflow_api_folder_path):
        self.comfyUI_root_path = comfyUI_root_path
        self.input_folder_path = os.path.join(self.comfyUI_root_path, COMFYUI_INPUT_FOLDER_RELATIVE_PATH)
        self.style_imgs_folder_path = os.path.join(self.input_folder_path, STYLE_IMG_FOLDER_NAME)
        os.makedirs(self.style_imgs_folder_path, exist_ok=True)
        self.workflow_api_folder_path = workflow_api_folder_path
        self.load_API_files()
        self.parse_all_API_data()

    def load_API_files(self):
        self.all_api_data = {}  # {API File Name: API Json Data, }
        for filename in os.listdir(self.workflow_api_folder_path):
            if filename.endswith('.json') and filename in AI_TEXTURING_WORKFLOW_API_NAMES:
                with open(os.path.join(self.workflow_api_folder_path, filename), 'rb') as file:
                    self.all_api_data[filename] = json.load(file)
                    
    class NodeType(enum.Enum):
        PARAMS = 0
        IMGS = 1
                    
    class NodeVarWrapper:
        def __init__(self, api_filename, node_title, node_id, node_class_type, node_inputs):
            self.api_filename = api_filename
            # remove the tag [Var] & [Order: int] and white space before them, also remove the white space at beginning or end of node title
            self.var_name = re.sub(r"\A\s*|\s*\[Var\]|\s*\[Imgs\]|\s*\[Order:[\D]*[\d]+\]|\s*\Z", "", node_title)
            order_match = re.search(r"\[Order:[\D]*([\d]+)\]", node_title)
            self.order = float('inf') if order_match is None else int(order_match.group(1))
            self.id = node_id
            self.class_type = node_class_type
            self.node_id_raw = str(node_id) + api_filename
            extra_tag_match = re.search(r"\s*(\[[\w\s:]*\])+\s*", self.var_name)
            self.node_class_name = node_class_type + "" if extra_tag_match is None else extra_tag_match.group(0)
            self.params = {}    # {Parameters Name: Property Name, }
            self.child_params = {}  # {Parameters Name: [Child Property Name, ], }
            
            if '[Imgs]' in node_title and self.class_type == "LoadImage":
                self.node_type = ComfyUIAPIHandler.NodeType.IMGS
                self.image_param_name = 'image'
                self.set_img_prop(node_inputs)
            else:
                self.node_type = ComfyUIAPIHandler.NodeType.PARAMS
                self.set_native_props(node_inputs)
        
        def get_prop_name(self, param_name):
            # Maximum property name length in Blender is 63, so we use hash of parameter path to create unique property name 
            prop_name_raw = param_name + self.node_id_raw
            return "NodeProp_" + hashlib.shake_128(str.encode(prop_name_raw)).hexdigest(8)
        
        def set_img_prop(self, node_inputs):
            
            img_prop_name = self.get_prop_name(self.image_param_name)
            
            if img_prop_name not in bpy.data.textures:
                texture = bpy.data.textures.new(img_prop_name, 'IMAGE')
                texture.image = bpy.data.images.new(img_prop_name, width=512, height=512)
                
            self.params[self.image_param_name] = img_prop_name
            self.child_params[self.image_param_name] = []
            
            self.img_prop_name = img_prop_name
            self.img_default_filename = node_inputs[self.image_param_name]
        
        def set_native_props(self, node_inputs):
            # Create properties for all parameters of this node
            for param_name in node_inputs:
                param_value = node_inputs[param_name]
                if type(param_value) in TYPE_2_PROP:
                    self.set_prop(param_name, param_value, TYPE_2_PROP[type(param_value)])
                    self.child_params[param_name] = []
        
        def set_prop(self, param_name, param_value, PropType):
            prop_name = self.get_prop_name(param_name)
            
            if not hasattr(bpy.context.scene, prop_name):
            
                param_prop = PropType(name=prop_name, 
                                    default=param_value,
                                    update=lambda scene, context: sync_child_nodes_properties(scene, context, self, param_name))

                setattr(bpy.types.Scene, prop_name, param_prop)
                
            self.params[param_name] = prop_name
            
        def set_nodes_to_sync(self, first_node_to_sync, last_node_to_sync):
            """
            Args:
                first_node_to_sync (NodeVarWrapper): The first node from previous stages that has same name & type as this node
                last_node_to_sync (NodeVarWrapper): The last node from previous stages that has same name & type as this node
            """
            self.now_nodes_to_sync_index = 0
            if first_node_to_sync is not None:
                self.nodes_to_sync = [self, first_node_to_sync, last_node_to_sync]   # [None, First Node to Sync, last Node to Sync]
                
                # change now_nodes_to_sync_index accordingly if parameters value in this node is equal to parameters value in self.nodes_to_sync
                scene = bpy.context.scene
                for i in range(1, len(self.nodes_to_sync)):
                    node_to_sync = self.nodes_to_sync[i]
                    for param_name in self.params:
                        if ( (hasattr(scene, self.params[param_name]) and hasattr(scene, node_to_sync.params[param_name]) and 
                            getattr(scene, self.params[param_name]) != getattr(scene, node_to_sync.params[param_name]))
                            or
                            (self.node_type == ComfyUIAPIHandler.NodeType.IMGS and node_to_sync.node_type == ComfyUIAPIHandler.NodeType.IMGS and
                            self.img_default_filename != node_to_sync.img_default_filename) ):
                            break
                    else:
                        self.now_nodes_to_sync_index = i
                        break
                
                # setup synchronize EnumProperty for this node
                self.node_sync_enum_prop_name = "NodeSyncEnum_" + hashlib.shake_128(str.encode(self.node_id_raw)).hexdigest(8)
                node_sync_enum_prop = EnumProperty(
                    items=( (SYNC_TYPES[i], SYNC_TYPES[i], SYNC_TYPES_DESCRIPTION[i]) for i in range(len(SYNC_TYPES)) ), 
                    name=self.node_sync_enum_prop_name,
                    default=SYNC_TYPES[self.now_nodes_to_sync_index],
                    update=lambda scene, context: set_parent_node_to_sync(scene, context, self)
                )
                
                setattr(bpy.types.Scene, self.node_sync_enum_prop_name, node_sync_enum_prop)
                
                set_parent_node_to_sync(scene, bpy.context, self)
            else:
                self.nodes_to_sync = self.node_sync_enum_prop_name = None
    
    def parse_all_API_data(self):
        # Get all variable nodes from api data
        first_node_vars = {}
        last_node_vars = {}
        self.all_api_var = {}  # {API File Name: {Node Title: NodeVarWrapper}, }
        for filename in AI_TEXTURING_WORKFLOW_API_NAMES:
            tmp_first_node_vars = {}
            tmp_last_node_vars = {}
            api_var = {}
            
            api_data = self.all_api_data[filename]
            for node_id in api_data:
                node_data = api_data[node_id]
                node_title = node_data['_meta']['title']
                if '[Var]' in node_title:
                    
                    node_var = ComfyUIAPIHandler.NodeVarWrapper(filename, node_title, node_id, node_data['class_type'], node_data['inputs'])
                    
                    if node_var.node_class_name in first_node_vars:
                        first_node_to_sync = first_node_vars[node_var.node_class_name]
                        last_node_to_sync = last_node_vars[node_var.node_class_name]
                    else:
                        tmp_first_node_vars[node_var.node_class_name] = node_var
                        first_node_to_sync = last_node_to_sync = None
                        
                    node_var.set_nodes_to_sync(first_node_to_sync, last_node_to_sync)
                    
                    tmp_last_node_vars[node_var.node_class_name] = node_var
                    
                    api_var[node_title] = node_var
                    
            self.all_api_var[filename] = api_var
            first_node_vars.update(tmp_first_node_vars)
            last_node_vars.update(tmp_last_node_vars)
        
        #print(f"self.all_api_var: {self.all_api_var}")
        
        # Sort the variable nodes
        self.all_node_var_sorted = {} # {API File Name: [Node Title]}
        for filename in self.all_api_var:
            api_var = self.all_api_var[filename]
    
            self.all_node_var_sorted[filename] = sorted(api_var.keys(), key=lambda node_title:api_var[node_title].order)
            
        #print(f"self.all_node_var_sorted: {self.all_node_var_sorted}")
            
    def sync_workflow_api_data(self, filename):
        # synchronize all api data from UI properties should be called before send api data to ComfyUI servers
        if filename in self.all_api_var:
            api_var = self.all_api_var[filename]
            api_data = self.all_api_data[filename]
            
            scene = bpy.context.scene
            
            for node_title in api_var:
                node_var = api_var[node_title]
                node_var_params = node_var.params
                node_data_params = api_data[node_var.id]['inputs']
                
                if node_var.node_type == ComfyUIAPIHandler.NodeType.IMGS:
                    img_prop_name = node_var_params[node_var.image_param_name]
                    img = bpy.data.textures[img_prop_name].image
                    folder, filename = os.path.split(img.filepath)
                    
                    input_img_filepath = os.path.join(self.style_imgs_folder_path, filename)
                    if not os.path.exists(input_img_filepath): 
                        img.save(filepath=input_img_filepath, quality=100)
                        
                    node_data_params[node_var.image_param_name] = os.path.join(STYLE_IMG_FOLDER_NAME, filename)
                else:
                    for param_name in node_var_params:
                        if hasattr(scene, node_var_params[param_name]):
                            node_data_params[param_name] = getattr(scene, node_var_params[param_name])
                    
        print(f"self.all_api_data: {self.all_api_data}")

             
def force_redraw():
    """
        A Blender panel is redrawn every time the panel needs to be updated, 
        such as when the user interacts with the panel or when the Blender application window is resized
    """
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

class VIEW3D_OT_ImportPath(Operator):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "import_func.path"  # important since its how bpy.ops.import_func.path is constructed
    bl_label = "Load Path"

    filepath: StringProperty(
        name="File Path",
        description="Filepath used for importing the file",
        maxlen=1024,
        subtype='FILE_PATH',
    )
    
    target_path_name: StringProperty(name="Target Path Name")

    def execute(self, context):
        folder, file = os.path.split(self.filepath)
        print(folder, file)
        setattr(context.scene, self.target_path_name, self.filepath) 
        return {'FINISHED'}

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
class VIEW3D_OT_CallComfyUIAPI(Operator):
    bl_idname = "import_func.api"
    bl_label = "Handle ComfyUI's API"
    
    api_fliename: StringProperty(name="Workflow API File's Name")
    
    def execute(self, context):
        comfyUIAPIHandler.sync_workflow_api_data(self.api_fliename)
        return {'FINISHED'}

class TextImg2TexturePanel:
    # where to add the panel in the UI
    bl_space_type = "VIEW_3D"  # 3D Viewport area
    bl_region_type = "UI"  # Sidebar region
    bl_category = "BlenderAI43D"  # found in the Sidebar

class VIEW3D_PT_TextImg2Texture(Panel, TextImg2TexturePanel):  # class naming convention ‘CATEGORY_PT_name’
    bl_label = "Text/Image to UV Texture"  # found at the top of the Panel
    
    def draw(self, context): # called every time the panel is redrawn
        layout = self.layout
        scene = context.scene
        
        # path input
        row = layout.row()
        row.prop(scene, WORKFLOW_FOLDER_PATH_PROP['name'], text=WORKFLOW_FOLDER_PATH_PROP['name'])
        op = row.operator(VIEW3D_OT_ImportPath.bl_idname, text="Load Path")
        op.target_path_name = WORKFLOW_FOLDER_PATH_PROP['name']
        
        # float input
        row = layout.row()
        row.prop(scene, CFG_PROP['name'], text=CFG_PROP['name'])
        
        # enum input
        row = layout.row()
        row.prop(scene, CHECKPOINT_PROP['name'], text=CHECKPOINT_PROP['name'])
    
class TextImg2TextureSubPanel(TextImg2TexturePanel):
    bl_parent_id = "VIEW3D_PT_TextImg2Texture"
    
class WorkflowAPIUILoader(Panel, TextImg2TextureSubPanel):
    api_filename = ""

    def __init__(self, api_filename_index=0):
        super().__init__()
        self.api_filename = AI_TEXTURING_WORKFLOW_API_NAMES[api_filename_index]

    def load_vars(self, layout, context):
        scene = context.scene

        for node_title in comfyUIAPIHandler.all_node_var_sorted[self.api_filename]:
            node_var = comfyUIAPIHandler.all_api_var[self.api_filename][node_title]
            
            # Draw Node Title
            row = layout.row()
            row.label(text=node_var.var_name)
            if node_var.node_sync_enum_prop_name is not None:
                row.prop(scene, node_var.node_sync_enum_prop_name, text="")
                
            if node_var.node_type == ComfyUIAPIHandler.NodeType.IMGS:
                # Draw Image selection Window
                row = layout.row()
                img_prop_name = node_var.params[node_var.image_param_name]
                row.template_ID_preview(bpy.data.textures[img_prop_name], "image", open="image.open")
                
            elif node_var.now_nodes_to_sync_index == 0:
                box = layout.box()
                # Draw Node Parameters
                node_var_params = node_var.params
                for param_name in node_var_params:
                    row = box.row()
                    row.prop(scene, node_var_params[param_name], text=param_name)
                    
            layout.separator(factor=1)
        
class VIEW3D_PT_MVImagesGeneration(WorkflowAPIUILoader):
    bl_label = "Step 1: Multi-view Images Generation"
    
    def __init__(self):
        super().__init__(api_filename_index=0)

    def draw(self, context):
        layout = self.layout
        
        row = layout.row()
        
        if REF_STYLE_IMG_NAME in bpy.data.images:
            texture = bpy.data.textures[REF_STYLE_IMG_NAME]
            row.template_ID_preview(texture, "image", open="image.open")
            print(texture.image.filepath) # absolute path
            
        self.load_vars(layout, context)
        
        row = layout.row()
        row.operator(VIEW3D_OT_CallComfyUIAPI.bl_idname, text="Draft")
        row = layout.row()
        op = row.operator(VIEW3D_OT_CallComfyUIAPI.bl_idname, text="Generate")
        op.api_fliename = self.api_filename
    
class VIEW3D_PT_FaceDetailer(WorkflowAPIUILoader):
    bl_label = "Step 1.5 (Optional): Face Detailer"

    def __init__(self):
        super().__init__(api_filename_index=1)

    def draw(self, context):
        layout = self.layout
        
        self.load_vars(layout, context)
        
        row = layout.row()
        row.operator(VIEW3D_OT_CallComfyUIAPI.bl_idname, text="Draft")
        row = layout.row()
        op = row.operator(VIEW3D_OT_CallComfyUIAPI.bl_idname, text="Face Detailing All")
        op.api_fliename = self.api_filename

class VIEW3D_PT_UpscaleMVImages(WorkflowAPIUILoader):
    bl_label = "Step 2: Upscale generated Multi-view Images"

    def __init__(self):
        super().__init__(api_filename_index=2)

    def draw(self, context):
        layout = self.layout
        
        self.load_vars(layout, context)
        
        row = layout.row()
        row.operator(VIEW3D_OT_CallComfyUIAPI.bl_idname, text="Draft")
        row = layout.row()
        op = row.operator(VIEW3D_OT_CallComfyUIAPI.bl_idname, text="Upscale All")
        op.api_fliename = self.api_filename
        
class VIEW3D_PT_VFI(WorkflowAPIUILoader):
    bl_label = "Step 2.5: VFI for Upscaled Images"

    def __init__(self):
        super().__init__(api_filename_index=3)

    def draw(self, context):
        layout = self.layout
        
        self.load_vars(layout, context)
        
        row = layout.row()
        op = row.operator(VIEW3D_OT_CallComfyUIAPI.bl_idname, text="Video Frame Interpolation")
        op.api_fliename = self.api_filename
        
class VIEW3D_PT_BakeTexture(WorkflowAPIUILoader):
    bl_label = "Step 3 (Optional): Bake Result to Texture"

    @classmethod
    def calc_before_draw(cls):
        # Calculate here before draw
        pass

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator("mesh.primitive_cube_add", text="Bake UV Texture")
    
        
class VIEW3D_PT_TextImg2Mesh(Panel):  # class naming convention ‘CATEGORY_PT_name’

    # where to add the panel in the UI
    bl_space_type = "VIEW_3D"  # 3D Viewport area
    bl_region_type = "UI"  # Sidebar region

    bl_category = "BlenderAI43D"  # found in the Sidebar
    bl_label = "Text/Image to 3D Mesh"  # found at the top of the Panel

    def draw(self, context):
        """define the layout of the panel"""
        layout = self.layout
        row = layout.row()
        row.label(text="Coming Soon!!!")

ot_classes = (
    VIEW3D_OT_ImportPath,
    VIEW3D_OT_CallComfyUIAPI,
)

pt_classes = (
    VIEW3D_PT_TextImg2Texture,
    VIEW3D_PT_MVImagesGeneration,
    VIEW3D_PT_FaceDetailer,
    VIEW3D_PT_UpscaleMVImages,
    VIEW3D_PT_VFI,
    VIEW3D_PT_BakeTexture,
    VIEW3D_PT_TextImg2Mesh,
)

def register():
    from bpy.utils import register_class
    
    for cls in ot_classes:
        register_class(cls)
        
    for cls in pt_classes:
        register_class(cls)
        
    for prop in all_props:
        setattr(bpy.types.Scene, prop['name'], prop['prop'])
    
    for img_name in all_imgs_names:
        texture = bpy.data.textures.new(img_name, 'IMAGE')
        texture.image = bpy.data.images.new(img_name, width=512, height=512)

def unregister():
    from bpy.utils import unregister_class
    
    # Worked for now, but there should be a better way to do so...
    for pt in Panel.__subclasses__():
        if pt.bl_space_type == 'VIEW_3D' and any(str(cls) in str(pt) for cls in pt_classes):
            try:
                unregister_class(pt)
            except:
                continue
            
    for ot in Operator.__subclasses__():
        if any(str(cls) in str(ot) for cls in ot_classes):
            try:
                unregister_class(ot)
            except:
                continue

    for prop in all_props:
            delattr(bpy.types.Scene, prop['name'])
    
    for img_name in all_imgs_names:
        bpy.data.textures.remove(bpy.data.textures[img_name])
        bpy.data.images.remove(bpy.data.images[img_name])

if __name__ == "__main__":
    #unregister()
    comfyUIAPIHandler = ComfyUIAPIHandler("C:/Users/reall/Softwares/ComfyUI_windows_portable", "C:/Users/reall/Softwares/Miniconda3/envs/AITexturing/_Projects/dreamgaussian/blender_py/APIs/")
    register()