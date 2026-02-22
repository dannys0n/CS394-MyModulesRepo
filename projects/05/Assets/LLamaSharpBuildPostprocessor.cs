#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEditor.Callbacks;
using System.IO;

public class LLamaSharpBuildPostprocessor {

    /// <summary>
    ///  Due to how LLamaSharp looks for the .dll files, we need to copy them to the project directory where exe is from plugins directory
    /// </summary>
    /// <param name="target"></param>
    /// <param name="pathToBuiltProject"></param>
    [PostProcessBuild(1)]
    public static void OnPostprocessBuild(BuildTarget target, string pathToBuiltProject) {
        string pluginsDirectory;
        pathToBuiltProject = Path.GetDirectoryName(pathToBuiltProject);
        if (target == BuildTarget.StandaloneWindows64)
        {
            pluginsDirectory = Path.Join(
                Path.Join(
                    pathToBuiltProject, $"{PlayerSettings.productName}_Data", "Plugins"
                ), 
                "x86_64"
            );
        }
        else if (target == BuildTarget.StandaloneWindows)
        {
            pluginsDirectory = Path.Join(
                Path.Join(
                    pathToBuiltProject, $"{PlayerSettings.productName}_Data", "Plugins"
                ), 
                "x86"
            );
        }
        else
        {
            Debug.LogError("Unsupported build target");
            return;
        }

        var nativeDlls = new[]
        {
            "llama.dll",
            "libllama.dll",
            "cudart64_12.dll",
            "cublas64_12.dll",
            "cublasLt64_12.dll"
        };

        var copiedBackend = false;
        foreach (var dllName in nativeDlls)
        {
            var sourcePath = Path.Join(pluginsDirectory, dllName);
            if (!File.Exists(sourcePath))
            {
                continue;
            }

            var destinationPath = Path.Join(pathToBuiltProject, dllName);
            File.Copy(sourcePath, destinationPath, true);
            Debug.Log($"Copied native runtime: {dllName}");

            if (dllName == "llama.dll" || dllName == "libllama.dll")
            {
                copiedBackend = true;
            }
        }

        if (!copiedBackend)
        {
            Debug.LogError("No LLama backend DLL was found in the built Plugins directory.");
        }
    }
}
#endif
