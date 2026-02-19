import sys
import json5
import jsbeautifier


def check_and_format_json5(filename):
    # read raw text (comments preserved)
    with open(filename, "r") as f:
        raw = f.read()

    # ---------- validate JSON5 ----------
    try:
        json5.loads(raw)
    except Exception as e:
        print("❌ JSON5 syntax error:")
        print(e)
        return False

    # ---------- beautify (keeps comments!) ----------
    opts = jsbeautifier.default_options()
    opts.indent_size = 4
    opts.preserve_newlines = True

    formatted = jsbeautifier.beautify(raw, opts)

    # overwrite file
    with open(filename, "w") as f:
        f.write(formatted)

    print("✅ JSON5 valid. Reformatted in place.")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python format_json5.py file.json5")
        sys.exit(1)

    check_and_format_json5(sys.argv[1])
