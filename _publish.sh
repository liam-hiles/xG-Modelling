# _publish.sh — A shortcut to save and publish your work to GitHub.
#
# Usage (in your R terminal):
#   ./_publish.sh "Your commit message here"
#
# Example:
#   ./_publish.sh " "

git add .
git commit -m "$1"
git push -u origin main

echo "Done! Your work has been published to GitHub."