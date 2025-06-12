
# HOOK LEADER TCL
# This code runs on each leader rank,
#      i.e., once per node.

# Set a root data directory
set root $env(HOME)/data
puts "HOOK HOST: [exec hostname]"

# Get the leader communicator from ADLB
set comm [ adlb::comm_get leaders ]
# Get my rank among the leaders
set rank [ adlb::comm_rank $comm ]

if { $rank == 0 } {
  set start [ clock seconds ]
}

# If I am rank=0, discover all files to bcast:
if { $rank == 0 } {
  set pattern $env(HOME)/proj/I-UNO.jw/exp_result/*.parquet
  puts "pattern: $pattern"
  set files [ glob $pattern ]
  puts "files: $files"
}

# Broadcast the file list to all leaders
turbine::c::bcast $comm 0 files

# Make a node-local data directory
set LOCAL /tmp/$env(USER)
file mkdir $LOCAL

# Copy each file to the node-local directory
foreach f $files {
  if { $rank == 0 } {
    puts "copying: $f"
  }
  turbine::c::copy_to $comm $f $LOCAL
}

if { $rank == 0 } {
  set stop [ clock seconds ]
  puts "hook time: [ expr $stop - $start ]"
}
