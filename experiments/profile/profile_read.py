import pstats
import sys

if __name__ == '__main__':
    
    profile_name = sys.argv[1]

    p = pstats.Stats(profile_name)
    p.sort_stats('tottime')
    p.print_stats(20)
