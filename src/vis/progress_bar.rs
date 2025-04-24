use std::sync::Mutex;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use lazy_static::lazy_static;

pub type ProgressBarCallback = fn(usize, f64, f64);

lazy_static! {
    static ref PROGRESS_BAR: Mutex<Option<ProgressBar>> = Mutex::new(None);
    static ref START_TIME: Mutex<Option<Instant>> = Mutex::new(None);
    static ref LAST_EPOCH_GROUP: Mutex<usize> = Mutex::new(0);
    static ref TRAINING_EPOCHS: Mutex<usize> = Mutex::new(0);
}

pub fn init_progress_bar(epochs: usize) {
    // Initialize the progress bar with the number of epochs
    let mut progress_bar = PROGRESS_BAR.lock().unwrap();
    *TRAINING_EPOCHS.lock().unwrap() = epochs;
    
    if progress_bar.is_none() {
        *progress_bar = Some(ProgressBar::new(epochs as u64));
        if let Some(pb) = &*progress_bar {
            pb.set_style(ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} epochs (cost: {msg})")
                .unwrap()
                .progress_chars("##-"));
        }
    }
}

// Create a progress bar callback
pub fn progress_bar_callback(epoch: usize, cost: f64, perc: f64) {
    // Static variables to maintain state between calls
    
    // Initialize on first call
    let progress_bar = PROGRESS_BAR.lock().unwrap();
    if progress_bar.is_none() {
        panic!("Progress bar not initialized. Call init_progress_bar() first.");
    }
    
    if let Some(pb) = &*progress_bar {
        // Update the progress bar
        pb.set_position(epoch as u64);
        pb.set_message(format!("{:.6}", cost));
        
        // Create a new line every 10 epochs
        let current_group = epoch / 10;
        let mut last_epoch_group = LAST_EPOCH_GROUP.lock().unwrap();
        if current_group > *last_epoch_group {
            *last_epoch_group = current_group;
            pb.println(format!("Completed epoch {} - Cost: {:.6}", current_group, cost));
        }
    }
}
