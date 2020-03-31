# dsl-compile --py [path/to/python/file] --output [path/to/output/tar.gz]

import kfp
from kfp import dsl

def get_new_images_op():
    return dsl.ContainerOp(
        name='get_new_images',
        image='k4rth33k/new_images',
        # command=['sh', '-c'],
        # arguments=['echo "hello world"']
    )

def train_op():
    return dsl.ContainerOp(
        name='train',
        image='k4rth33k/train',
        # command=['sh', '-c'],
        # arguments=['echo "hello world"']
    )

@dsl.pipeline(
    name='COVID',
    description='COVID-19 Transfer Learning Model Automation Pipeline.'
)
def covid_pipeline():
    _images_task = get_new_images_op()
    _train_task = train_op().after(_images_task)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(covid_pipeline, __file__ + '.yaml')