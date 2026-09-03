# =============================================================================
# Native AOT build of a .NET 10 file-based app.
#
# Two stages, and the discard is what actually keeps the image small: the SDK
# layer is ~2GB and never ships. The nuget cache clear keeps the build layer
# honest for anyone who inspects it, but it is the stage boundary that does the
# real work.
#
# The final image carries no .NET runtime at all — Native AOT produces a
# self-contained native binary, so runtime-deps (just the native libraries
# libc/openssl/icu need) is the whole dependency surface.
# =============================================================================

FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build

# Native AOT shells out to a real linker. Without clang and zlib headers the
# publish fails at the link step, after a full compile — a slow way to find out.
RUN apt-get update \
 && apt-get install -y --no-install-recommends clang zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY bot.cs .

# All build configuration lives in the `#:property` directives at the top of
# bot.cs, including PublishAot and the trim feature switches. Nothing is passed
# here, so the file is the single source of truth for how it is built.
RUN dotnet publish bot.cs -o /out \
 && dotnet nuget locals all --clear \
 && rm -rf /root/.nuget /tmp/*

# -----------------------------------------------------------------------------
# runtime-deps chiselled: no shell, no package manager, no .NET runtime, and a
# non-root default user. There is nothing in here to exec into, which is the
# point for a process holding an admin credential.
# -----------------------------------------------------------------------------
FROM mcr.microsoft.com/dotnet/runtime-deps:10.0-noble-chiseled AS final

WORKDIR /app
COPY --from=build /out/ ./

ENV ASPNETCORE_URLS=http://0.0.0.0:8080 \
    DOTNET_gcServer=0

EXPOSE 8080
ENTRYPOINT ["/app/bot"]
