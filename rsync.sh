rsync -avzP --update * ericmjl@rous:~/Documents/github/protein-convolutional-nets --exclude-from rsync_exclude.txt

rsync -avzP --update ericmjl@rous:~/Documents/github/protein-convolutional-nets/* ./ --exclude-from rsync_exclude.txt
