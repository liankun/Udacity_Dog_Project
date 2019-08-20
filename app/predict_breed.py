import argparse


parser = argparse.ArgumentParser()

parser.add_argument('input',action='store',help='image path')
parser.add_argument('checkpoint',action='store',help='model checkpoint')
parser.add_argument('--top_k',action='store',default=5,help='top k prediction',type=int)
parser.add_argument('--category_names',action='store',default=None,help='category names file')
parser.add_argument('--gpu',action='store_true',default=False,help='enable gpu')
parser.add_argument('--verbose',action='store_true',default=False)

result = parser.parse_args()
if result.verbose:
    print("image:          {!r}".format(result.input))
    print("checkpoint:     {!r}".format(result.checkpoint))
    print("top k :         {!r}".format(result.top_k))
    print("category names: {!r}".format(result.category_names))
    print("enable gpu:     {!r}".format(result.gpu))
