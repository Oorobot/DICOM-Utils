import numpy as np
import vtk


# -----------------------------------------------------------#
#                       自定义键盘交互
# -----------------------------------------------------------#
class KeyPressStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self):
        super().__init__()
        self.AddObserver("KeyPressEvent", self.OnKeyPress)

    def OnKeyPress(self, obj, event):
        print(f"Key {obj.GetInteractor().GetKeySym()} pressed")
        camera = (
            obj.GetInteractor()
            .GetRenderWindow()
            .GetRenderers()
            .GetFirstRenderer()
            .GetActiveCamera()
        )
        if obj.GetInteractor().GetKeySym() == "w":
            direction_of_projection = camera.GetDirectionOfProjection()
            position = camera.GetPosition()
            new_position = np.array(position) + 100 * np.array(direction_of_projection)
            camera.SetPosition(*new_position.tolist())
        if obj.GetInteractor().GetKeySym() == "s":
            direction_of_projection = camera.GetDirectionOfProjection()
            position = camera.GetPosition()
            new_position = np.array(position) - 100 * np.array(direction_of_projection)
            camera.SetPosition(*new_position.tolist())


# -----------------------------------------------------------#
#                   索引号 --> 实际坐标位置
# -----------------------------------------------------------#
def index_to_physical_position(boxes, origin, spacing):
    physical_positions = []
    origin = np.array(origin)
    spacing = np.array(spacing)
    for box in boxes:
        point1, point2 = np.array(box[0:3]), np.array(box[3:6])
        x1, y1, z1 = origin + point1 * spacing
        x2, y2, z2 = origin + point2 * spacing
        physical_positions.append(
            [
                [x1, y1, z1],
                [x1, y2, z1],
                [x2, y2, z1],
                [x2, y1, z1],
                [x1, y1, z2],
                [x1, y2, z2],
                [x2, y2, z2],
                [x2, y1, z2],
            ]
        )
    return physical_positions
