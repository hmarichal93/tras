import streamlit as st
import git
import os

def pull_last_changes_from_remote_repo(REPO_DIR):
    try:
        # Open the local repository
        repo = git.Repo(REPO_DIR)

        # Fetch updates from the remote repository
        origin = repo.remotes.origin
        origin.fetch()  # Fetches the latest changes from the remote

        # Get the latest local and remote commits on the main branch
        local_commit = repo.commit('main')
        remote_commit = repo.commit('origin/main')

        # Compare local and remote commits to see if there are updates
        if local_commit != remote_commit:
            st.warning("Changes detected in the remote main branch. Updating...")

            # Pull updates to the local repository
            origin.pull('main')
            #delete old config
            os.system("rm -rf ./config/runtime.json")

            st.success("Repository updated successfully.")

            # Reload the app to apply the changes
            st.rerun()
        else:
            st.info("The local repository is up-to-date with the remote main branch.")
    except Exception as e:
        st.error(f"Error checking or updating the repository: {e}")

    return