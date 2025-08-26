@ECHO OFF

pushd .

cd %~dp0

set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
if errorlevel 1 exit /b 1

popd
