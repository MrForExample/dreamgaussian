import bpy

#Operator settings
class SimpleOperatorPropertyGroup(bpy.types.PropertyGroup):
    location: bpy.props.FloatVectorProperty(name="Location", size=3)
    hidden: bpy.props.BoolProperty(name="Hidden")


#Draw layout for operator and panel
class OperatorDrawSettings:
    def draw(self, context):
        prop_grp = context.window_manager.simple_operator
        
        layout = self.layout
        col = layout.column()
        col.prop(prop_grp, "location")
        col.prop(prop_grp, "hidden")

        
#Common properties for panels
class View3DPanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tool"
    
    @classmethod
    def poll(cls, context):
        return (context.object is not None)

        
#Panel for operator                
class VIEW3D_PT_custom_redo_panel(View3DPanel, bpy.types.Panel):
    bl_label = "Custom Redo Panel"

    def draw(self, context):
        layout = self.layout
        col = layout.column()        
        col.operator("object.simple_operator", text="Execute Simple Operator")


#Sub panel for operator settings        
class VIEW3D_PT_custom_operator_settings_panel(View3DPanel, OperatorDrawSettings, bpy.types.Panel):
    bl_parent_id = "VIEW3D_PT_custom_redo_panel"    
    bl_label = "Simple Operator Settings"

    
#Object simple operator       
class OBJECT_OT_simple_operator(OperatorDrawSettings, bpy.types.Operator):
    bl_idname = "object.simple_operator"
    bl_label = "Simple Object Operator"
    bl_options = {'REGISTER', 'UNDO'}
    
        
    def execute(self, context):
        prop_grp = context.window_manager.simple_operator        
                
        context.object.location = prop_grp.location
        context.object.hide_set(prop_grp.hidden)
        return {'FINISHED'}


classes = (
    SimpleOperatorPropertyGroup,
    VIEW3D_PT_custom_redo_panel,
    VIEW3D_PT_custom_operator_settings_panel,    
    OBJECT_OT_simple_operator,          					 
)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    bpy.types.WindowManager.simple_operator = bpy.props.PointerProperty(type=SimpleOperatorPropertyGroup)

    
def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
        
    del bpy.types.WindowManager.simple_operator        

    
if __name__ == "__main__":
    register()