from pydantic import BaseModel

class VectorMetadata(BaseModel):
    name: str

class Vector(BaseModel):
    id: str
    values: list[float]
    metadata: VectorMetadata

class Point(BaseModel):
    x: int
    y: int

class Box(BaseModel):
    top_left: Point
    bottom_right: Point