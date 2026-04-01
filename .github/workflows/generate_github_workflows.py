from pathlib import Path
import jinja2


def main():
    """
    Generates GitHub workflow YAML files from Jinja2 templates.
    """
    workflows_dir = Path(__file__).parent
    print(f"Scan dir: {workflows_dir}")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(workflows_dir)),
        block_start_string="<%",
        block_end_string="%>",
        variable_start_string="<<",
        variable_end_string=">>",
    )

    for template_path in workflows_dir.glob("*.yml.j2"):
        template = env.get_template(template_path.name)
        content = template.render()

        yaml_path = template_path.with_suffix("")
        with open(yaml_path, "w") as f:
            f.write(
                "#" * 80
                + "\n# This file is auto-generated from the .j2 file via generate_github_workflows.py. Do not edit manually.\n"
                + "#" * 80
                + "\n"
            )
            f.write(content)

        print(f"Generated {yaml_path} from {template_path}")


if __name__ == "__main__":
    main()
