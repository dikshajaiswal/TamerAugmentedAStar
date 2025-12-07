def Findinstallationorder(packages, dependencies):
    """
    Finds the installation order for packages respecting dependencies.
    Args:
      packages: dict mapping package name -> install time
      dependencies: list of (dependent, dependency, type) where type = 'M' or 'O'
    Returns:
      list of package names in install order, or None if circular dependency
    """
    from collections import defaultdict

    # Build graph for mandatory dependencies
    graph_M = defaultdict(list)
    indegree = {pkg: 0 for pkg in packages}

    # Optional dependencies: dependent -> set of optional deps
    optional_deps = defaultdict(set)

    for dependent, dependency, dep_type in dependencies:
        if dep_type == 'M':
            graph_M[dependency].append(dependent)
            indegree[dependent] += 1
        else:
            optional_deps[dependent].add(dependency)

    ready = {pkg for pkg, deg in indegree.items() if deg == 0}
    installed = []
    installed_set = set()

    while ready:
        # Preferred: packages with all optional deps satisfied
        satisfied = []
        for pkg in ready:
            if optional_deps[pkg].issubset(installed_set):
                satisfied.append(pkg)

        if satisfied:
            candidates = satisfied
        else:
            candidates = list(ready)

        # Choose best candidate:
        # 1. lowest install time
        # 2. alphabetical order
        best_pkg = min(candidates, key=lambda p: (packages[p], p))

        ready.remove(best_pkg)
        installed.append(best_pkg)
        installed_set.add(best_pkg)

        # Reduce mandatory indegree
        for dep in graph_M[best_pkg]:
            indegree[dep] -= 1
            if indegree[dep] == 0:
                ready.add(dep)

    # If not all packages installed → circular dependency
    if len(installed) != len(packages):
        return None

    return installed


# ------------------------------------------------------------------------------
# MAIN: Input / Output handling
# ------------------------------------------------------------------------------

def main():
    import sys

    data = sys.stdin.read().strip().split()
    idx = 0

    # Number of packages
    n = int(data[idx]); idx += 1

    packages = {}
    for _ in range(n):
        name = data[idx]; idx += 1
        time = int(data[idx]); idx += 1
        packages[name] = time

    # Number of dependencies
    m = int(data[idx]); idx += 1
    dependencies = []

    for _ in range(m):
        dependent = data[idx]; idx += 1
        dependency = data[idx]; idx += 1
        dep_type = data[idx]; idx += 1  # "M" or "O"
        dependencies.append((dependent, dependency, dep_type))

    # Compute order
    order = Findinstallationorder(packages, dependencies)

    if order is None:
        print("Error: Circular dependency detected")
    else:
        for pkg in order:
            print(f"Installing {pkg}")


# Run the program
if __name__ == "__main__":
    main()
