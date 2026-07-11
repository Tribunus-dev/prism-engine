use std::io::{BufRead, BufReader, BufWriter, Write};
use std::net::Shutdown;
use std::os::unix::net::UnixStream;

/// socket→stdout on two independent threads.
pub fn run_proxy(socket_path: &str) -> anyhow::Result<()> {
    let stream = UnixStream::connect(socket_path)?;
    let reader = stream.try_clone()?;
    let writer = stream;

    let stdin_handle = {
        let writer = writer.try_clone()?;
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut buf = BufReader::new(stdin.lock());
            let mut out = BufWriter::new(writer);
            let mut line = String::new();
            while matches!(buf.read_line(&mut line), Ok(n) if n > 0) {
                if line.trim().is_empty() {
                    line.clear();
                    continue;
                }
                if writeln!(out, "{}", line.trim()).is_err() {
                    break;
                }
                out.flush().ok();
                line.clear();
            }
            let _ = out.flush();
            let _ = out.get_ref().shutdown(Shutdown::Write);
        })
    };

    let stdout_handle = {
        let reader = reader.try_clone()?;
        std::thread::spawn(move || {
            let mut buf = BufReader::new(reader);
            let stdout = std::io::stdout();
            let mut out = BufWriter::new(stdout.lock());
            let mut line = String::new();
            while matches!(buf.read_line(&mut line), Ok(n) if n > 0) {
                if line.trim().is_empty() {
                    line.clear();
                    continue;
                }
                if writeln!(out, "{}", line.trim()).is_err() {
                    break;
                }
                out.flush().ok();
                line.clear();
            }
        })
    };

    stdin_handle.join().ok();
    let _ = reader.shutdown(Shutdown::Both);
    stdout_handle.join().ok();
    Ok(())
}
