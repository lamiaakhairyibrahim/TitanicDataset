
import argparse
from data_coll import Collect

def call_class(path_data):
    opj = Collect(path_data)


def main():
    pars = argparse.ArgumentParser(description="enter the argument path of data")
    pars.add_argument('path' , help= 'the path to incloude')
    args = pars.parse_args()


    try:
        call_class(args.path )
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()