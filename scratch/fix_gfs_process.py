import sys
import re

content = open("src/nwpblend/ingest/gfs.py", "r").read()

# 1. Move do_fetch to module level
if "def do_fetch(" not in content:
    sys.exit(0)

# Find do_fetch body
do_fetch_pattern = r"    def do_fetch\(idx_p, vars_list, s3_p, out_file\):(.*?)(?=    for lead in leads:)"
match = re.search(do_fetch_pattern, content, re.DOTALL)
if not match:
    print("Could not find do_fetch")
    sys.exit(1)

do_fetch_body = match.group(0)

# Unindent do_fetch body by 4 spaces
unindented_body = "\n".join([line[4:] if line.startswith("    ") else line for line in do_fetch_body.splitlines()])

# Add import s3fs inside do_fetch since it will run in a separate process
unindented_body = unindented_body.replace(
    "def do_fetch(idx_p, vars_list, s3_p, out_file):",
    "def do_fetch(idx_p, vars_list, s3_p, out_file):\n    import s3fs\n    fs = s3fs.S3FileSystem(anon=True, config_kwargs={'read_timeout': 15, 'connect_timeout': 5})\n"
)

# Remove the nested do_fetch from the original content
content = content.replace(do_fetch_body, "")

# Add unindented_body to the top of the file (after imports)
import_idx = content.find("logger = logging.getLogger(__name__)")
content = content[:import_idx] + "logger = logging.getLogger(__name__)\n\n" + unindented_body + "\n\n" + content[import_idx + len("logger = logging.getLogger(__name__)"):]

# 2. Replace ThreadPoolExecutor with multiprocessing.Process
old_exec = """                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    future = executor.submit(do_fetch, idx_path, gfs_vars, s3_path, target_file)
                    future.result(timeout=download_timeout)
                    time.sleep(1)
                except Exception as e:
                    if isinstance(e, concurrent.futures.TimeoutError):"""

new_exec = """                    import multiprocessing
                    p = multiprocessing.Process(target=do_fetch, args=(idx_path, gfs_vars, s3_path, target_file))
                    p.start()
                    p.join(timeout=download_timeout)
                    if p.is_alive():
                        p.terminate()
                        p.join()
                        raise TimeoutError(f"Download timed out after {download_timeout}s")
                    if p.exitcode != 0:
                        raise RuntimeError(f"Subprocess failed with exit code {p.exitcode}")
                    time.sleep(1)
                except Exception as e:
                    if isinstance(e, TimeoutError):"""

content = content.replace(old_exec, new_exec)

with open("src/nwpblend/ingest/gfs.py", "w") as f:
    f.write(content)
