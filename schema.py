from pydantic import BaseModel

class Point(BaseModel):
    x: int
    y: int

class Box(BaseModel):
    top_left: Point
    bottom_right: Point