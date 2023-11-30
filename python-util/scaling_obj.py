# code adapted from https://blender.stackexchange.com/a/251578

def rescale_obj(obj_path, obj_scaled_path, scale):
    with open(obj_path, 'r') as source:
        with open(obj_scaled_path, 'w') as target:
            for line in source:
                taget_line = line

                if(line.startswith('v ')):
                    coordinates = [float(coordinate) for coordinate in line.split(' ')[1:]]
                    rescaled = [c*scale for c in coordinates]
                    rescaled_as_str = " ".join([str(c) for c in rescaled])
                    taget_line = f'v {rescaled_as_str}\n'

                target.write(taget_line)

def main():
    rescale_obj("bunny.obj", "bunny_scaled.obj", 10.0)

if __name__ == '__main__':
    main()