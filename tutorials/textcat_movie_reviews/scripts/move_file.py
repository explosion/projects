import argparse
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='The path to the input file to move into another directory.')
    parser.add_argument('-o', '--output', required=True,
                        help='The path to the output directory where the input file must be moved.')
    args = parser.parse_args()
    print(args)

    print('pre')
    shutil.move(args.input, '{}/{}'.format(args.output, args.input.split('/')[-1]))
    print('post')

if __name__ == '__main__':
    main()