import { Trash2, Download, FileBox } from 'lucide-react';
import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { deleteFile, getAllFiles } from '../helper/db';
import toast from 'react-hot-toast';

const RecentFiles = () => {
  const [generatedFiles, setRecentFiles] = useState(null);

  const base64ToBlob = (base64Data, contentType = 'model/gltf-binary') => {
    const byteCharacters = atob(base64Data.split(',')[1]);
    const byteArrays = [];
    for (let offset = 0; offset < byteCharacters.length; offset += 512) {
      const slice = byteCharacters.slice(offset, offset + 512);
      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      byteArrays.push(byteArray);
    }
    return new Blob(byteArrays, { type: contentType });
  };

  const handleRedownload = (data) => {
    if (!data) {
      console.error("No saved data found to redownload.");
      return;
    }

    const { fileData, filename = "download.glb" } = JSON.parse(data);
    const blob = base64ToBlob(fileData);
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    link.parentNode.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleRemove = (id) => {
    const confirmed = window.confirm("Are you sure you want to delete this?");
    if (confirmed) {
      try {
        deleteFile(id);
        setRecentFiles(generatedFiles.filter(item => item.id !== id));
        toast.success('Successfully Removed! 🗑️');
      } catch (error) {
        toast.error('Failed to delete file');
      }
    }
  };

  useEffect(() => {
    (async () => {
      const glbFiles = await getAllFiles();
      setRecentFiles(glbFiles);
    })();
  }, []);

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="text-center">
        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-white">
          Download your previously generated 3D models
        </h1>
      </section>

      {/* Files List */}
      {generatedFiles?.length > 0 && (
        <section className="glass p-0 overflow-hidden">
          <div className="p-5 sm:p-6 border-b border-white/10">
            <h3 className="text-lg sm:text-xl font-semibold text-white">Generated Files</h3>
          </div>
          
          <div className="divide-y divide-white/10">
            {generatedFiles.map((file) => (
              <div 
                key={file.id} 
                className="p-5 sm:p-6 flex items-center justify-between gap-4 hover:bg-white/5 transition-colors"
              >
                <div className="flex items-center gap-4 min-w-0">
                  <div className="h-12 w-12 shrink-0 rounded-xl bg-anim-badge flex items-center justify-center">
                    <Download className="h-5 w-5 text-white" />
                  </div>
                  <div className="min-w-0">
                    <h4 className="text-white font-medium text-sm sm:text-base truncate">
                      {file.filename}
                    </h4>
                  </div>
                </div>

                <div className="flex items-center gap-2 sm:gap-3 shrink-0">
                  <button
                    onClick={() => handleRedownload(file)}
                    className="rounded-xl bg-white/10 hover:bg-white/15 px-3 sm:px-5 py-2 text-sm font-medium text-white transition flex items-center gap-2"
                  >
                    <Download className="h-4 w-4" />
                    <span className="hidden sm:inline">Download</span>
                  </button>
                  <button
                    onClick={() => handleRemove(file.id)}
                    className="rounded-xl bg-red-500/20 hover:bg-red-500/30 px-3 sm:px-5 py-2 text-sm font-medium text-red-300 transition flex items-center gap-2"
                  >
                    <Trash2 className="h-4 w-4" />
                    <span className="hidden sm:inline">Delete</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Empty State */}
      {generatedFiles?.length === 0 && (
        <section className="glass p-10 sm:p-16 text-center">
          <FileBox className="h-16 w-16 text-white/40 mx-auto mb-4" />
          <h4 className="text-white font-medium text-lg mb-2">No recent files</h4>
          <p className="text-white/60 text-sm mb-6">
            Generate your first 3D model to see it here
          </p>
          <NavLink
            to="/"
            className="inline-flex items-center gap-2 rounded-xl bg-white/10 hover:bg-white/15 px-6 py-3 text-sm font-medium text-white transition"
          >
            Generate Now
          </NavLink>
        </section>
      )}

      {/* Loading State */}
      {generatedFiles === null && (
        <section className="glass p-10 sm:p-16 text-center">
          <div className="h-8 w-8 border-2 border-white/20 border-t-white rounded-full animate-spin mx-auto mb-4" />
          <p className="text-white/60 text-sm">Loading files...</p>
        </section>
      )}
    </div>
  );
};

export default RecentFiles;
